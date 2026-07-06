"""Per-character RAG memory and scratchpad."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from chromadb import Metadata
from chromadb.api.types import OneOrMany

from typing import TYPE_CHECKING

from ara.memory.chroma import ChromaStore
from ara.utils.logger import get_logger

if TYPE_CHECKING:
    from ara.llm.client import LLMClient
    from ara.world.character import Character

logger = get_logger(__name__)

_DEPTH_MAP = {
    'shallow': 2,
    'medium': 5,
    'deep': 10,
    'very_deep': 30,
}


@dataclass
class Scratchpad:
    """Ephemeral per-character notepad carried across scene boundaries.

    Characters can overwrite their scratchpad via the ``write_scratch`` tool.
    The previous scene's scratch is retained as ``prev_text`` so the model
    can reference it at the start of a new scene.
    """

    text: str = 'Nothing yet!'
    """Current scratch content."""

    prev_text: str = 'Nothing yet!'
    """Snapshot from the previous scene."""

    def prepare_for_new_scene(self) -> None:
        """Archive the current scratch and reset to the default."""
        self.prev_text = self.text
        self.text = 'Nothing yet!'


class NullMemory:
    """No-op memory backend for characters that do not persist conversations."""

    def add_conversation(self, texts: list[str]) -> None:
        """Do nothing."""

    def recall(self, queries: list[str], depth: str = 'medium') -> list[str]:
        """Return an empty list."""
        return []


@dataclass
class CharacterMemory:
    """Vector-backed conversational memory for a single character.

    Each character receives its own ChromaDB collection named after the
    character's stable UUID.
    """

    character_id: uuid.UUID
    """Stable identifier for the character."""

    db: ChromaStore
    """Shared ChromaDB store instance."""

    collection_name: str = field(init=False)
    """Resolved collection name (``str(character_id)``)."""

    def __post_init__(self) -> None:
        """Resolve the collection name from the character ID."""
        self.collection_name = str(self.character_id)

    def add_conversation(self, texts: list[str]) -> None:
        """Embed and store conversation snippets.

        :param texts: Snippets to persist into the character's collection.
        """
        if not texts:
            return
        logger.debug(
            f'Adding {len(texts)} conversation snippets to {self.collection_name}'
        )
        ids = [str(uuid.uuid4()) for _ in texts]
        metadatas: OneOrMany[Metadata] = [{'memory': True} for _ in texts]
        self.db.upsert(
            self.collection_name,
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

    def recall(
        self,
        queries: list[str],
        depth: str = 'medium',
        client: 'LLMClient | None' = None,
        querier: 'Character | None' = None,
    ) -> list[str]:
        """Retrieve semantically similar conversation snippets.

        :param queries: Search query strings.
        :param depth: Recall breadth-one of ``shallow``, ``medium``, ``deep``,
            or ``very_deep``.
        :param client: Optional LLM client for querier-aware filtering.
        :param querier: Optional character whose perspective should shape the answer.
        :return: Flattened list of retrieved document strings.
        """
        logger.debug(
            f'Recalling from {self.collection_name}: {queries!r} (depth={depth})'
        )
        n_results = _DEPTH_MAP.get(depth, 5)
        results = self.db.query(
            self.collection_name,
            query_texts=queries,
            n_results=n_results,
            where={'memory': {'$eq': True}},
        )
        docs = results.get('documents', [[]]) or []
        flat: list[str] = []
        for group in docs:
            flat.extend(group or [])

        if client is not None and querier is not None and flat:
            context = (
                f"Querier: {querier.name}\n"
                f"Personality/expertise: {querier.card_fields.get('personality', '')}\n"
                f"Scenario: {querier.card_fields.get('scenario', '')}\n"
                f"Query: {queries[0] if queries else ''}\n\n"
                f"Raw memories:\n" + "\n".join(f"- {d}" for d in flat)
            )
            filtered = client.complete_subagent(
                task=(
                    "Reframe the raw memories into an answer for the querier. "
                    "If the querier is a child, simplify. "
                    "If the querier is an expert, be detailed and precise. "
                    "You must base your answer ONLY on the raw memories provided. "
                    "If no memory answers the query, return exactly '-- nothing relevant found'. "
                    "Do not invent, infer, or hallucinate facts that are not present. "
                    "Return a short paragraph."
                ),
                context=context,
                max_tokens=256,
            )
            if filtered.strip():
                return [filtered.strip()]

        return flat
