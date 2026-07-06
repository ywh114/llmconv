"""Tests for the ChromaDB wrapper."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from ara.config import AraSettings
from ara.memory.chroma import ChromaStore


def test_clear_all_collections_removes_everything() -> None:
    """clear_all_collections deletes all existing collections."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        settings = AraSettings(
            data_dir=tmp,
            api_key="",
            api_endpoint="",
            api_model="",
        )
        db = ChromaStore(settings)

        db.upsert("story_history", ids=["h1"], documents=["old summary"])
        db.upsert(
            "orchestrator_wiki",
            ids=["w1"],
            documents=["old spoiler"],
            metadatas=[{"topic": "events"}],
        )
        char_id = uuid.uuid4()
        db.upsert(str(char_id), ids=["m1"], documents=["old memory"])

        assert {c.name for c in db.client.list_collections()} == {
            "story_history",
            "orchestrator_wiki",
            str(char_id),
        }

        db.clear_all_collections()

        assert db.client.list_collections() == []
