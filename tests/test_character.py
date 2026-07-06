"""Tests for :mod:`ara.world.character`."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ara.memory.chroma import ChromaStore
from ara.world.character import Importance
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

    def test_load_sprite_descriptions(self) -> None:
        """Per-sprite descriptions are parsed from meta.toml."""
        mock_db = MagicMock(spec=ChromaStore)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "TestChar"
            base.mkdir()
            (base / "card.toml").write_text(
                'name = "TestChar"\n'
                'summary = ""\npersonality = ""\nscenario = ""\n'
                'greeting_message = ""\nexample_messages = ""\n'
            )
            (base / "meta.toml").write_text(
                'sprites = ["default_neutral", "work_neutral"]\n'
                'default_sprite = "default_neutral"\n\n'
                '[default_neutral]\n'
                'description = "A plain naval uniform."\n\n'
                '[work_neutral]\n'
                'description = "A crisp office uniform."\n'
            )
            (base / "default_neutral.png").write_bytes(b"")
            (base / "work_neutral.png").write_bytes(b"")

            char = load_character(base, mock_db)

        assert char.sprite_descriptions == {
            "default_neutral": "A plain naval uniform.",
            "work_neutral": "A crisp office uniform.",
        }
        assert char.skin_description() == "A plain naval uniform."
        assert char.skin_description("work_neutral") == "A crisp office uniform."
        assert char.skin_description("unknown") == ""
