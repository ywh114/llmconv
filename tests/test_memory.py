"""Tests for :mod:`ara.memory.knowledge`."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad


class TestScratchpad:
    """Tests for the ephemeral scratchpad."""

    def test_prepare_for_new_scene(self) -> None:
        """Preparing for a new scene should archive the current text."""
        pad = Scratchpad(text="Current plans")
        pad.prepare_for_new_scene()
        assert pad.prev_text == "Current plans"
        assert pad.text == "Nothing yet!"


class TestCharacterMemory:
    """Tests for vector-backed character memory."""

    def test_recall_flattening(self) -> None:
        """Recall should flatten nested document groups from ChromaDB."""
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.query.return_value = {
            "documents": [
                ["doc1", "doc2"],
                ["doc3"],
            ]
        }

        mem = CharacterMemory(
            character_id=uuid.uuid4(),
            db=mock_db,
        )
        results = mem.recall(["query"], depth="medium")

        assert results == ["doc1", "doc2", "doc3"]
        mock_db.query.assert_called_once()

    def test_add_conversation_skips_empty(self) -> None:
        """``add_conversation`` should be a no-op when given an empty list."""
        mock_db = MagicMock(spec=ChromaStore)
        mem = CharacterMemory(
            character_id=uuid.uuid4(),
            db=mock_db,
        )
        mem.add_conversation([])
        mock_db.upsert.assert_not_called()
