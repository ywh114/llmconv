"""Tests for :mod:`ara.world.character`."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ara.memory.chroma import ChromaStore
from ara.models import Importance
from ara.world.character import load_character


class TestLoadCharacter:
    """Tests for character loading from disk."""

    def test_load_from_card_toml(self) -> None:
        """Loading from ``card.toml`` should produce a valid Character."""
        mock_db = MagicMock(spec=ChromaStore)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "TestChar"
            base.mkdir()
            (base / "card.toml").write_text(
                'name = "TestChar"\n'
                'summary = "A test character"\n'
                'personality = "Cheerful"\n'
                'scenario = "Testing"\n'
                'greeting_message = "Hello!"\n'
                'example_messages = "Hi there"\n'
            )
            (base / "meta.toml").write_text('importance = "IMPORTANT"\n')

            char = load_character(base, mock_db)

        assert char.name == "TestChar"
        assert char.importance == Importance.IMPORTANT
        assert char.card_fields["personality"] == "Cheerful"

    def test_missing_assets_raises(self) -> None:
        """An empty directory should raise :exc:`RuntimeError`."""
        mock_db = MagicMock(spec=ChromaStore)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "Empty"
            base.mkdir()
            with pytest.raises(RuntimeError):
                load_character(base, mock_db)
