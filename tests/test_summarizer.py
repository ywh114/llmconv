"""Tests for summarizer state-modifier parsing and application."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.world.story import Story
from ara.world.summarizer import SceneStateModifiers, _extract_state_modifiers

from tests.helpers import make_scene


class TestExtractStateModifiers:
    def test_character_status_block(self) -> None:
        # Modifier blocks run to the next header or EOF, so the block comes
        # after the narrative text.
        text = (
            'Narrative text before.\n'
            'CHARACTER_STATUS Alice: {"title": "Worried", "sections": []}\n'
        )
        cleaned, modifiers = _extract_state_modifiers(text)
        assert modifiers.character_status == {
            "Alice": {"title": "Worried", "sections": []}
        }
        assert "CHARACTER_STATUS" not in cleaned
        assert "Narrative text before." in cleaned

    def test_character_status_block_non_dict_is_ignored(self) -> None:
        _, modifiers = _extract_state_modifiers('CHARACTER_STATUS Alice: "just a string"')
        assert modifiers.character_status == {}

    def test_character_status_alongside_other_kinds(self) -> None:
        text = (
            'PLAYER_STATUS: {"title": "P", "sections": []}\n'
            'CHARACTER_STATUS Alice: {"title": "A", "sections": []}\n'
            'LOCATION_STATUS Kitchen: {"title": "K", "sections": []}\n'
            'SPRITE Bob: {"sprite": "hidden", "visible_to": ["Alice"]}\n'
        )
        _, modifiers = _extract_state_modifiers(text)
        assert modifiers.player_status == {"title": "P", "sections": []}
        assert modifiers.character_status == {"Alice": {"title": "A", "sections": []}}
        assert modifiers.location_status == {"Kitchen": {"title": "K", "sections": []}}
        assert modifiers.sprites == {"Bob": {"sprite": "hidden", "visible_to": ["Alice"]}}

    def test_bare_status_passes_through(self) -> None:
        """Bare 'STATUS <Name>:' belongs to the transition parser's flag
        channel (character_status_updates), not to state modifiers — it must
        pass through untouched."""
        text = 'STATUS Bob:\n{"wounded": true}\n'
        cleaned, modifiers = _extract_state_modifiers(text)
        assert modifiers.character_status == {}
        # Only line-ending normalization; content passes through intact.
        assert cleaned == text.rstrip("\n")


class TestApplyCharacterStatusModifier:
    def test_character_status_applied_to_scene_character(self) -> None:
        """The full path: modifiers carry canonical character status into
        Character.status (normalize display names, then apply)."""
        mock_db = MagicMock(spec=ChromaStore)
        mock_client = MagicMock(spec=LLMClient)
        scene = make_scene("scene_a", mock_db, char_names=("Player", "Narrator", "Alice"))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
            (tmp / "scene_a.toml").write_text('id = "scene_a"\n')
            story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")

            modifiers = SceneStateModifiers()
            modifiers.character_status["Alice"] = {
                "title": "Worried",
                "sections": [{"type": "bars", "items": [{"label": "HP", "value": 5, "max": 10}]}],
            }
            normalized = story._normalize_state_modifiers(scene, modifiers)
            story._apply_state_modifiers(scene, normalized)

            alice = next(c for c in scene.character_pool if c.name == "Alice")
            assert alice.status["title"] == "Worried"

    def test_unknown_character_is_skipped(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_client = MagicMock(spec=LLMClient)
        scene = make_scene("scene_a", mock_db, char_names=("Player", "Narrator", "Alice"))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
            (tmp / "scene_a.toml").write_text('id = "scene_a"\n')
            story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")

            modifiers = SceneStateModifiers()
            modifiers.character_status["Ghost"] = {"title": "X", "sections": []}
            normalized = story._normalize_state_modifiers(scene, modifiers)
            assert normalized.character_status == {}
