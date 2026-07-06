"""Long-term story-memory retrieval for cross-scene consistency.

This module provides a thin wrapper around the ``story_history`` ChromaDB
collection so the summarizer (and optionally the orchestrator) can recall
relevant earlier scenes when bridging into a new scene.
"""

from __future__ import annotations

from ara.memory.chroma import ChromaStore
from ara.utils.logger import get_logger

logger = get_logger(__name__)


class StoryMemory:
    """Semantic recall over the story's scene-history collection."""

    def __init__(self, db: ChromaStore | None) -> None:
        """Create a story-memory helper.

        :param db: Shared ChromaDB store. If ``None``, recall always returns an
            empty list.
        """
        self.db = db

    def recall(
        self,
        queries: list[str],
        *,
        n_results: int = 3,
    ) -> list[str]:
        """Return the most relevant past scene summaries for the given queries.

        Results are deduplicated across queries and ordered by first appearance.

        :param queries: Query phrases. Typically the upcoming scene's plot,
            character names, or key story threads.
        :param n_results: Number of hits to fetch per query.
        :return: Flat list of scene-summary strings.
        """
        if not queries or self.db is None:
            return []

        try:
            result = self.db.query(
                "story_history",
                query_texts=queries,
                n_results=n_results,
            )
        except Exception as exc:
            logger.warning(f"StoryMemory recall query failed: {exc}")
            return []

        documents = result.get("documents") or []
        seen: set[str] = set()
        hits: list[str] = []
        for per_query in documents:
            if not per_query:
                continue
            for doc in per_query:
                if not doc or doc in seen:
                    continue
                seen.add(doc)
                hits.append(doc)
        return hits
