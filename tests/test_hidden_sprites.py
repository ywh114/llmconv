"""Tests for hidden sprites and visible_to."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.models import GameRole, StreamResult
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.persistence.save import SaveManager, SAVE_VERSION
from ara.world.character import Character, Importance
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene
from ara.world.story import Story


def _make_char(name: str, mock_db: ChromaStore) -> Character:
    cid = uuid.uuid5(uuid.NAMESPACE_DNS, f"test.{name}")
    return Character(
        id=cid,
        canonical_name=name,
        name=name,
        card_fields={
            "name": name,
            "summary": f"{name} summary",
            "personality": f"{name} personality",
            "scenario": f"{name} scenario",
            "greeting_message": f"Hi, I'm {name}",
            "example_messages": "",
        },
        importance=Importance.IMPORTANT,
        memory=CharacterMemory(character_id=cid, db=mock_db),
        scratch=Scratchpad(),
    )


def _make_scene(mock_db: ChromaStore) -> Scene:
    chars = {_make_char(name, mock_db) for name in ["Player", "Narrator", "Alice", "Bob"]}
    player = next(c for c in chars if c.name == "Player")
    narrator = next(c for c in chars if c.name == "Narrator")
    loc = Location(canonical_name="room", name="room", desc="A room.")
    return Scene(
        id="hidden_scene",
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool=chars,
        starting_characters=chars,
        player=player,
        narrator=narrator,
        location_pool={loc},
        starting_location=loc,
        plot_considerations="",
        plot_story="Test hidden sprites",
        next_choices={},
    )


def test_scene_load_hidden_sprite_with_visible_to() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cc = tmp / "assets" / "cc" / "hidden"
        for name in ("Player", "Narrator", "Alice", "Bob"):
            path = cc / name
            path.mkdir(parents=True)
            (path / "card.toml").write_text(
                f'name = "{name}"\n'
                'summary = ""\n'
                'personality = ""\n'
                'scenario = ""\n'
                'greeting_message = ""\n'
                'example_messages = ""\n',
                encoding="utf-8",
            )

        plot_dir = tmp / "assets" / "plot" / "hidden"
        plot_dir.mkdir(parents=True)
        scene_toml = plot_dir / "hidden_scene.toml"
        scene_toml.write_text(
            'id = "hidden_scene"\n'
            'name = "Hidden Scene"\n'
            'language = "English"\n'
            'zeitgeist = "test"\n'
            'tone = "neutral"\n'
            '\n'
            '[character]\n'
            'pool = ["Player", "Narrator", "Alice", "Bob"]\n'
            'inits = ["Player", "Narrator", "Alice", "Bob"]\n'
            'player = "Player"\n'
            'narrator = "Narrator"\n'
            '\n'
            '[character.sprites]\n'
            'Alice = { sprite = "hidden", visible_to = ["Bob"] }\n'
            '\n'
            '[location]\n'
            'pool = ["room"]\n'
            'init = "room"\n'
            '\n'
            '[location.descs]\n'
            'room = "A room."\n'
            '\n'
            '[plot]\n'
            'scene = "Test"\n',
            encoding="utf-8",
        )

        mock_db = MagicMock(spec=ChromaStore)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        scene = Scene.load(scene_toml, mock_db, config)

        alice = next(c for c in scene.character_pool if c.name == "Alice")
        assert alice.current_sprite == "none"
        assert alice.hidden is True
        assert alice.visible_to == {"Bob"}

        bob = next(c for c in scene.character_pool if c.name == "Bob")
        assert bob.hidden is False


def test_engine_filters_hidden_from_observer_context() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene(mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")
    bob = next(c for c in scene.character_pool if c.name == "Bob")

    # Alice is hidden; only the Player can see her, not Bob.
    alice.hidden = True
    alice.visible_to = {"Player"}

    engine = Engine(mock_client, db=None)
    engine.start(scene)

    # Seed the context with a public line from Alice.
    engine.ctx.assistant_message("Alice speaks.", tool_calls=[], name=alice.name)

    decision = TurnDecision(
        next_char=bob,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
    )
    mock_client.complete.return_value = StreamResult(
        content="Bob replies.", tool_calls=[], reasoning_content=""
    )

    output = engine._character_turn(scene, engine.ctx, decision, engine.loc)
    assert output == "Bob replies."
    # Bob's branch should not contain Alice's message because she is hidden from him.
    call_messages = mock_client.complete.call_args.kwargs["messages"]
    alice_messages_in_branch = [msg for msg in call_messages if msg.get("name") == alice.name]
    assert not alice_messages_in_branch


def test_save_load_preserves_hidden_state() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene(mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")
    alice.hidden = True
    alice.visible_to = {"Bob"}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "hidden_scene.toml").write_text('id = "hidden_scene"\nname = "Hidden Scene"\n')

        story = Story(config, mock_db, mock_client, tmp / "hidden_scene.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene

        manager = SaveManager(config)
        path = manager.save(story, slot=1)
        snapshot = json.loads(path.read_text())
        alice_data = next(c for c in snapshot["characters"] if c["name"] == "Alice")
        assert alice_data["hidden"] is True
        assert set(alice_data["visible_to"]) == {"Bob"}

        fresh_story = Story(config, mock_db, mock_client, tmp / "hidden_scene.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        fresh_alice = next(c for c in fresh_story.current_scene.character_pool if c.name == "Alice")
        assert fresh_alice.hidden is True
        assert fresh_alice.visible_to == {"Bob"}
