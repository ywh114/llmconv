"""Centralized wiki storage for world facts.

The wiki is a ChromaDB collection of world facts shared by the orchestrator,
characters, and the story runner.  :class:`WikiStore` owns all access to that
collection: semantic recall with trust annotation and querier-aware filtering,
entry writes and deletes, and bulk ingestion of world-setting files, narrative
state, and invented facts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ara.utils.logger import get_logger
from ara.world.setting import load_world_setting

if TYPE_CHECKING:
    from ara.llm.client import LLMClient
    from ara.memory.chroma import ChromaStore
    from ara.world.character import Character

logger = get_logger(__name__)


WIKI_COLLECTION = "orchestrator_wiki"


class WikiStore:
    """Single access point for the orchestrator wiki collection.

    :param db: ChromaDB store backing the wiki, or ``None`` to disable it.
    :param client: Optional LLM client used by the querier-aware filtering
        subagent during recall.
    :param collection_name: Name of the ChromaDB collection holding wiki
        entries.
    """

    def __init__(
        self,
        db: ChromaStore | None,
        client: LLMClient | None = None,
        collection_name: str = WIKI_COLLECTION,
    ) -> None:
        """Create the store."""
        self.db = db
        self.client = client
        self.collection_name = collection_name

    @staticmethod
    def normalize_doc(doc: str) -> str:
        """Return a canonical form of a wiki document for deduplication."""
        text = doc.strip()
        if text.startswith('-'):
            text = text[1:].strip()
        if text.startswith('(trust:') and ')' in text:
            text = text.split(')', 1)[1].strip()
        return ' '.join(text.split())

    def recall(
        self,
        query: str,
        n_results: int = 3,
        annotate_trust: bool = False,
        querier: Character | None = None,
        dedup_against: str = '',
        exclude_docs: set[str] | None = None,
        max_distance: float | None = 0.65,
    ) -> str:
        """Search the wiki for relevant entries.

        :param query: Search query.
        :param n_results: Maximum number of entries to retrieve.
        :param annotate_trust: When ``True``, prefix each result with its trust score.
        :param querier: Optional character requesting the information.  When
            provided, a filtering subagent reframes the result for the querier's
            perspective and expertise.
        :param dedup_against: Optional block of already-prefetched wiki text.
            Entries whose normalized content appears in it are filtered out.
        :param exclude_docs: Optional set of normalized document contents to
            omit from the result.  Used by the summarizer to avoid retrieving
            the same wiki entry for multiple prefetch queries.
        :param max_distance: Maximum ChromaDB distance for a result to be
            considered relevant.  Results farther than this are discarded.
        :return: Formatted wiki results or a failure message.
        """
        if self.db is None:
            return 'Wiki is not available.'
        logger.debug(f'Wiki recall query: {query!r}')
        try:
            results = self.db.query(
                self.collection_name,
                query_texts=[query],
                n_results=n_results,
            )
            docs = results.get('documents', [[]])[0] or []
            if not docs:
                return 'No relevant wiki entries found.'

            ids = results.get('ids', [[]])[0] or [
                str(i) for i in range(len(docs))
            ]
            metadatas = results.get('metadatas', [[]])[0] or [{} for _ in docs]
            distances = results.get('distances', [[]])[0] or []

            if max_distance is not None and distances:
                filtered = [
                    (d, i, m)
                    for d, i, m, dist in zip(docs, ids, metadatas, distances)
                    if dist <= max_distance
                ]
                if not filtered:
                    return 'No relevant wiki entries found.'
                docs, ids, metadatas = [list(x) for x in zip(*filtered)]

            items = list(zip(docs, ids, metadatas))
            if exclude_docs:
                items = [
                    (d, i, m) for d, i, m in items
                    if self.normalize_doc(d) not in exclude_docs
                ]
                if not items:
                    return 'All results already covered by existing context.'
            docs, ids, metadatas = [list(x) for x in zip(*items)]

            if annotate_trust or querier is not None:
                if annotate_trust:
                    formatted: list[str] = []
                    for doc, meta in zip(docs, metadatas):
                        trust = (
                            meta.get('trust')
                            if isinstance(meta, dict)
                            else None
                        )
                        if trust is not None:
                            formatted.append(f'(trust: {trust}) {doc}')
                        else:
                            formatted.append(f'{doc}')
                    docs = formatted

                if querier is not None:
                    docs = self.filter_for_querier(
                        query, docs, ids, metadatas, querier
                    )

            if dedup_against:
                prefetched_docs = {
                    self.normalize_doc(chunk)
                    for chunk in dedup_against.split('\n\n')
                    if chunk.strip()
                }
                docs = [
                    d for d in docs
                    if self.normalize_doc(d) not in prefetched_docs
                ]

            if not docs:
                return 'All results already covered by existing context.'

            text = '\n\n'.join(f'- {d}' for d in docs)
            logger.debug(f'Wiki recall result ({len(docs)} docs):\n{text}')
            return text
        except Exception as exc:
            logger.debug(f'Wiki recall failed: {exc}')
            return 'Wiki recall failed.'

    def filter_for_querier(
        self,
        query: str,
        raw_docs: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]],
        querier: Character,
    ) -> list[str]:
        """Run retrieved wiki documents through a querier-aware subagent.

        The subagent reframes facts (or invents consistent answers when no raw
        document matches) based on the querier's identity, age, and expertise.
        Topic IDs and source-world metadata are provided so the subagent can
        resolve conflicts between the initial world setting and later settings.
        """
        if not raw_docs:
            return raw_docs
        doc_lines = []
        for doc_id, meta, doc in zip(ids, metadatas, raw_docs):
            world = (
                meta.get('world', 'unknown')
                if isinstance(meta, dict)
                else 'unknown'
            )
            topic = (
                meta.get('topic', doc_id) if isinstance(meta, dict) else doc_id
            )
            doc_lines.append(f'[{topic} | world: {world}] {doc}')
        context = (
            f'Querier: {querier.name}\n'
            f'Personality/expertise: {querier.card_fields.get("personality", "")}\n'
            f'Scenario/background: {querier.card_fields.get("scenario", "")}\n'
            f'Original query: {query}\n\n'
            f'Raw wiki documents (topic ID and source world shown in brackets):\n'
            + '\n'.join(f'- {line}' for line in doc_lines)
        )
        filtered = self.client.complete_subagent(
            task=(
                'Reframe the raw wiki documents into an answer for the querier. '
                'If the querier is a child, simplify. '
                'If the querier is an expert, be detailed and precise. '
                'You must base your answer ONLY on the raw wiki documents provided. '
                'If documents conflict (for example, the initial world setting versus '
                'a mid-game setting), use the topic ID and source world metadata to '
                'decide which fact applies to the current query context. '
                "If no document answers the query, return exactly '-- nothing relevant found'. "
                'Do not invent, infer, or hallucinate facts that are not present. '
                'Return a short paragraph.'
            ),
            context=context,
            max_tokens=256,
        )
        if filtered.strip():
            return [filtered.strip()]
        return raw_docs

    def write(
        self,
        topic: str,
        content: str,
        importance: str = 'notable',
        trust: float = 0.0,
    ) -> str:
        """Write or overwrite a wiki entry.

        :param trust: Reliability score from -1.0 (explicit lie) to 1.0 (canon).
        """
        if self.db is None:
            return 'Wiki is not available.'
        try:
            self.db.upsert(
                self.collection_name,
                ids=[topic],
                documents=[content],
                metadatas=[
                    {'topic': topic, 'importance': importance, 'trust': trust}
                ],
            )
            result = f"Wiki entry '{topic}' saved."
            logger.debug(f'Wiki write: {result}')
            return result
        except Exception as exc:
            logger.debug(f'Wiki write failed: {exc}')
            return 'Wiki write failed.'

    def forget(self, doc_id: str) -> str:
        """Delete a wiki entry."""
        if self.db is None:
            return 'Wiki is not available.'
        try:
            collection = self.db.collection(self.collection_name)
            collection.delete(ids=[doc_id])
            result = f"Wiki entry '{doc_id}' deleted."
            logger.debug(f'Wiki forget: {result}')
            return result
        except Exception as exc:
            logger.debug(f'Wiki forget failed: {exc}')
            return 'Wiki forget failed.'

    def has_content(self) -> bool:
        """Return ``True`` if the wiki collection has documents."""
        if self.db is None:
            return False
        try:
            return self.db.collection(self.collection_name).count() > 0
        except Exception as exc:
            logger.debug(f"Could not check orchestrator_wiki content: {exc}")
            return False

    def ingest_setting_file(
        self, path: Path | None, label: str = "setting"
    ) -> None:
        """Load a single world-setting TOML and upsert its entries into the wiki."""
        if not path or not path.exists():
            logger.debug(f"No {label} found at {path}; skipping wiki upsert.")
            return

        try:
            setting = load_world_setting(path)
        except Exception as exc:
            logger.warning(f"Failed to load {label} from {path}: {exc}")
            return

        if self.db is None:
            return

        entries = setting.wiki_entries()
        if not entries:
            logger.debug(f"{label} '{setting.id}' has no wiki entries; skipping upsert.")
            return

        ids = list(entries.keys())
        docs = [entries[i] for i in ids]
        metadatas = [
            {"topic": topic, "importance": "critical", "world": setting.id}
            for topic in ids
        ]
        try:
            self.db.upsert(
                self.collection_name, ids=ids, documents=docs, metadatas=metadatas
            )
            logger.info(f"Loaded {label} '{setting.id}' with {len(ids)} wiki entries")
        except Exception as exc:
            logger.warning(f"Failed to upsert {label} into wiki: {exc}")

    def ingest_narrative_state(self, state: dict[str, Any]) -> None:
        """Mirror the story-level narrative state into the wiki."""
        if self.db is None or not state:
            return
        try:
            self.db.upsert(
                self.collection_name,
                ids=["story:state"],
                documents=[json.dumps(state, ensure_ascii=False)],
                metadatas=[{"topic": "story:state", "importance": "critical", "trust": 1.0}],
            )
            logger.debug("Mirrored narrative state to orchestrator_wiki")
        except Exception as exc:
            logger.warning(f"Failed to mirror narrative state: {exc}")

    def ingest_invented_facts(self, facts: list[dict[str, Any]]) -> None:
        """Persist invented facts from the summarizer into the wiki."""
        if not facts or self.db is None:
            return
        for idx, fact in enumerate(facts):
            statement = fact.get("fact", "").strip()
            if not statement:
                continue
            trust = float(fact.get("trust", 0.0))
            source = fact.get("source", "").strip()
            topic = f"invented_fact_{idx:03d}"
            content = statement
            if source:
                content += f"\nSource: {source}"
            try:
                self.db.upsert(
                    self.collection_name,
                    ids=[topic],
                    documents=[content],
                    metadatas=[{"topic": topic, "importance": "notable", "trust": trust}],
                )
                logger.info(f"Upserted invented fact '{topic}' with trust {trust}")
            except Exception as exc:
                logger.warning(f"Failed to upsert invented fact: {exc}")
