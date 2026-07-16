"""Tests for the save/load snapshot system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.persistence.save import SaveManager, SAVE_VERSION
from ara.world.character import create_anonymous_character
from ara.world.scene import Scene
from ara.world.story import Story

from tests.helpers import make_scene

_CHARS = ("Player", "Narrator", "Alice", "Bob")


def _make_scene(scene_id: str, mock_db: ChromaStore) -> Scene:
    return make_scene(scene_id, mock_db, char_names=_CHARS)


def test_save_load_round_trip_preserves_new_state() -> None:
    """Save/load preserves player_status, character status, and narrative state."""
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")
    bob = next(c for c in scene.character_pool if c.name == "Bob")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "scene_a.toml").write_text('id = "scene_a"\n')
        story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")

        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        # Simulate the scene having been loaded by a step().
        story._current_scene = scene

        # Mutate state that tickets 15-18 introduced.
        story.engine._player_status = {
            "title": "Status",
            "sections": [
                {"type": "inventory", "items": ["Key"]},
                {"type": "bars", "items": [{"label": "HP", "value": 85, "max": 100}]},
            ],
        }
        story.engine._world_time = "evening"
        story.engine._mechanical_changelog = [
            {"turn": 0, "type": "system_changes", "changes": {"inventory": ["Key"]}},
        ]
        story.engine._story_state = {"zombie_outbreak": True}
        story._character_status = {"Bob": {"wounded": True, "location": "infirmary"}}
        bob.status = dict(story._character_status["Bob"])  # simulates _load_scene applying registry
        story._narrative_state = {"port_alert_level": "red"}
        story._world_id = "azur_lane"
        story._world_loaded = True
        alice.scratch.text = "[Thought]: find the key"
        alice.scratch.prev_text = "[Thought]: old plan"
        alice.status = {"has_key": True}
        alice.current_sprite = "happy"
        scene.starting_location.desc = "A ransacked room."

        manager = SaveManager(config)
        path = manager.save(story, slot=1)
        assert path.exists()

        # Verify snapshot version.
        snapshot = json.loads(path.read_text())
        assert snapshot["version"] == SAVE_VERSION

        # Create a fresh story and load the snapshot.
        fresh_story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        assert fresh_story.engine._player_status == {
            "title": "Status",
            "sections": [
                {"type": "inventory", "items": ["Key"]},
                {"type": "bars", "items": [{"label": "HP", "value": 85, "max": 100}]},
            ],
        }
        assert fresh_story.engine._world_time == "evening"
        assert fresh_story.engine._mechanical_changelog == [
            {"turn": 0, "type": "system_changes", "changes": {"inventory": ["Key"]}},
        ]
        assert fresh_story.engine._story_state == {"zombie_outbreak": True}
        assert fresh_story._character_status == {"Bob": {"wounded": True, "location": "infirmary"}}
        assert fresh_story._narrative_state == {"port_alert_level": "red"}
        assert fresh_story._world_id == "azur_lane"
        assert "azur_lane" in fresh_story._loaded_settings

        # Character state restored on the loaded scene.
        loaded_alice = next(c for c in fresh_story.current_scene.character_pool if c.name == "Alice")
        assert loaded_alice.scratch.text == "[Thought]: find the key"
        assert loaded_alice.scratch.prev_text == "[Thought]: old plan"
        assert loaded_alice.status == {"has_key": True}
        assert loaded_alice.current_sprite == "happy"
        assert fresh_story.current_scene.starting_location.desc == "A ransacked room."

        # Off-screen character status is in the registry even if not in scene.
        loaded_bob = next(c for c in fresh_story.current_scene.character_pool if c.name == "Bob")
        assert loaded_bob.status == {"wounded": True, "location": "infirmary"}


def test_save_load_round_trip_preserves_language() -> None:
    """Save/load preserves the story's current language."""
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "scene_a.toml").write_text('id = "scene_a"\nlanguage = "zh"\n')
        story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")

        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene
        story._language = "zh"

        manager = SaveManager(config)
        path = manager.save(story, slot=1)
        snapshot = json.loads(path.read_text())
        assert snapshot["story_state"]["_language"] == "zh"

        fresh_story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)
        assert fresh_story.language == "zh"


def test_load_rejects_old_save_version() -> None:
    """Save files with an outdated version are rejected."""
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "scene_a.toml").write_text('id = "scene_a"\n')
        story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        story._current_scene = scene
        manager = SaveManager(config)
        manager.save(story, slot=1)

        path = manager.base_dir / tmp.name / "slot_01.json"
        data = json.loads(path.read_text())
        data["version"] = 1
        path.write_text(json.dumps(data))

        fresh_story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            with pytest.raises(ValueError, match="Unsupported save version 1"):
                manager.load(fresh_story, slot=1)


def _empty_db() -> MagicMock:
    """Return a mock ChromaStore whose collections are empty."""
    mock_db = MagicMock(spec=ChromaStore)
    mock_db.get_all.return_value = {"ids": [], "documents": [], "metadatas": []}
    return mock_db


def test_save_load_preserves_canonical_progress() -> None:
    """Canonical-script index and pending choices survive a save/load cycle."""
    mock_db = _empty_db()
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "scene_a.toml").write_text('id = "scene_a"\n')
        story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")

        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene

        story.engine._canonical_index = 3
        story.engine._canonical_pending_choices = [
            {"text": "Go left", "next_scene": "left"},
            {"text": "Go right", "next_scene": "right"},
        ]

        manager = SaveManager(config)
        manager.save(story, slot=1)

        fresh_story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        assert fresh_story.engine._canonical_index == 3
        assert fresh_story.engine._canonical_pending_choices == [
            {"text": "Go left", "next_scene": "left"},
            {"text": "Go right", "next_scene": "right"},
        ]


def test_save_load_preserves_card_overrides() -> None:
    """Transition-summarizer card overrides survive a save/load cycle."""
    mock_db = _empty_db()
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "scene_a.toml").write_text('id = "scene_a"\n')
        story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")

        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene

        alice.card_overrides = {"personality": "Injured and grumpy"}

        manager = SaveManager(config)
        manager.save(story, slot=1)

        fresh_story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        loaded_alice = next(
            c for c in fresh_story.current_scene.character_pool if c.name == "Alice"
        )
        assert loaded_alice.card_overrides == {"personality": "Injured and grumpy"}


def test_save_load_preserves_anonymous_characters() -> None:
    """Runtime-spawned anonymous NPCs survive a save/load cycle."""
    mock_db = _empty_db()
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)

    # _make_scene reuses the same set for pool and starting characters, so make
    # a distinct starting set before adding runtime extras.
    scene.starting_characters = set(scene.starting_characters)

    servant = create_anonymous_character(
        name="Servant", description="A nervous servant", sprite="servant_neutral"
    )
    servant.status = {"mood": "nervous"}
    guard = create_anonymous_character(name="Guard", description="A guard")
    guard.current_sprite = "guard_alert"
    scene.character_pool.add(servant)
    scene.character_pool.add(guard)
    scene.starting_characters.add(servant)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "scene_a.toml").write_text('id = "scene_a"\n')
        story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")

        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene

        story.engine._here_chars = set(scene.starting_characters)
        story.engine._away_chars = {guard}
        story.engine._loc = scene.starting_location
        story.engine._running = True

        manager = SaveManager(config)
        manager.save(story, slot=1)

        fresh_story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        loaded_scene = fresh_story.current_scene
        names = {c.name for c in loaded_scene.character_pool}
        assert "Servant" in names
        assert "Guard" in names

        here_names = {c.name for c in fresh_story.engine._here_chars}
        away_names = {c.name for c in fresh_story.engine._away_chars}
        assert "Servant" in here_names
        assert "Guard" in away_names
        assert "Guard" not in here_names

        loaded_servant = next(c for c in loaded_scene.character_pool if c.name == "Servant")
        assert loaded_servant.status == {"mood": "nervous"}
        loaded_guard = next(c for c in loaded_scene.character_pool if c.name == "Guard")
        assert loaded_guard.current_sprite == "guard_alert"


def test_save_load_preserves_transition_state() -> None:
    """A snapshot taken during the finalizing transition resumes at the next scene."""
    mock_db = _empty_db()
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "scene_a.toml").write_text('id = "scene_a"\n')
        (tmp / "scene_b.toml").write_text('id = "scene_b"\n')
        story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")

        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene

        story._state = "finalizing"
        story._current_path = tmp / "scene_b.toml"
        story._finalize_turn_text = "The sun sets over the gate."
        story._finalize_turn_changes = {"location": "Gate", "next_scene": "scene_b"}
        story.engine._next_scene = "scene_b"
        story.engine._running = False

        manager = SaveManager(config)
        manager.save(story, slot=1)

        fresh_story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        assert fresh_story._state == "finalizing"
        assert fresh_story._current_path == tmp / "scene_b.toml"
        assert fresh_story._finalize_turn_text == "The sun sets over the gate."
        assert fresh_story._finalize_turn_changes == {
            "location": "Gate",
            "next_scene": "scene_b",
        }
        assert fresh_story.engine._next_scene == "scene_b"
