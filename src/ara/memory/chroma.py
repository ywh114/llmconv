"""Thin wrapper around ChromaDB persistent client."""

from __future__ import annotations

from typing import Any

from chromadb import Metadata, PersistentClient, QueryResult
from chromadb.api.types import OneOrMany
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from ara.config import AraSettings
from ara.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaStore:
    """Manages ChromaDB collections with a configurable sentence-transformer
    embedding function.

    :param settings: Application settings used to resolve the persistent path
        and embedding model name.
    """

    def __init__(self, settings: AraSettings) -> None:
        """Create the store.

        :param settings: Application settings used to resolve the persistent
            path and embedding model name.
        """
        self.settings = settings
        self.client = PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model,
            token=False,
        )

    def collection(self, name: str):
        """Get or create a collection.

        :param name: Collection identifier.
        :return: A ChromaDB collection object.
        """
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self._ef,  # pyright: ignore [reportArgumentType]
        )

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        metadatas: OneOrMany[Metadata] | None = None,
    ) -> None:
        """Upsert documents into a collection.

        :param collection_name: Target collection.
        :param ids: Unique identifiers parallel to *documents*.
        :param documents: Text payloads to embed and store.
        :param metadatas: Optional metadata dicts parallel to *documents*.
        """
        logger.debug(f'Upsert {len(documents)} docs into {collection_name}')
        coll = self.collection(collection_name)
        coll.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 5,
        where: dict | None = None,
    ) -> QueryResult:
        """Query a collection by semantic similarity.

        :param collection_name: Target collection.
        :param query_texts: Query strings.
        :param n_results: Maximum results per query.
        :param where: Optional ChromaDB metadata filter.
        :return: Raw ChromaDB query result dict.
        """
        coll = self.collection(collection_name)
        return coll.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
        )

    def get_all(
        self,
        collection_name: str,
        where: dict | None = None,
    ) -> dict[str, Any]:
        """Retrieve all documents from a collection.

        :param collection_name: Target collection.
        :param where: Optional ChromaDB metadata filter.
        :return: Dict with ``ids``, ``documents``, and ``metadatas`` keys.
        """
        coll = self.collection(collection_name)
        return coll.get(where=where)  # type: ignore[return-value]

    def clear_all_collections(self) -> None:
        """Delete every existing ChromaDB collection.

        This makes the vector store fully transient. Callers are responsible
        for repopulating collections from a snapshot or world settings
        afterwards.
        """
        try:
            collections = list(self.client.list_collections())
        except Exception as exc:
            logger.warning(f"Could not list Chroma collections: {exc}")
            return

        for coll in collections:
            try:
                self.client.delete_collection(coll.name)
                logger.debug(f"Deleted Chroma collection {coll.name}")
            except Exception as exc:
                logger.warning(f"Could not delete Chroma collection {coll.name}: {exc}")
