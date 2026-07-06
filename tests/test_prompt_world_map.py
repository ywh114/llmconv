"""Tests for prompt tweaks and open-world textual map."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.world.character import Character, Importance
from ara.world.engine import _character_system_prompt
from ara.world.orchestrator import Orchestrator
from ara.world.scene import Location, Scene


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


def test_scene_load_world_map() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        story = "world_map_test"
        cc = tmp / "assets" / "cc" / story
        plot = tmp / "assets" / "plot" / story
        plot.mkdir(parents=True)
        for name in ("Player", "Narrator"):
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

        scene_toml = plot / "world_map_scene.toml"
        scene_toml.write_text(
            'id = "world_map_scene"\n'
            'name = "World Map Scene"\n'
            'language = "English"\n'
            '\n'
            '[character]\n'
            'pool = ["Player", "Narrator"]\n'
            'inits = ["Player", "Narrator"]\n'
            'player = "Player"\n'
            'narrator = "Narrator"\n'
            '\n'
            '[location]\n'
            'pool = ["room"]\n'
            'init = "room"\n'
            '\n'
            '[location.descs]\n'
            'room = "A room."\n'
            '\n'
            '[world]\n'
            'map = """\n'
            '玄城位于东州中部，北接云雾谷。\n'
            '"""\n'
            '\n'
            '[plot]\n'
            'scene = "Test"\n',
            encoding="utf-8",
        )

        mock_db = MagicMock(spec=ChromaStore)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        scene = Scene.load(scene_toml, mock_db, config)

        assert "玄城" in scene.world_map
        assert "World map" in scene.plot_as_tool_content()


def test_orchestrator_prompt_has_randomness_and_anonymity() -> None:
    mock_client = MagicMock(spec=LLMClient)
    orch = Orchestrator(mock_client, db=None)
    mock_db = MagicMock(spec=ChromaStore)
    player = _make_char("Player", mock_db)
    narrator = _make_char("Narrator", mock_db)
    scene = Scene(
        id="prompt_scene",
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool={player, narrator},
        starting_characters={player, narrator},
        player=player,
        narrator=narrator,
        location_pool={Location(canonical_name="room", name="room", desc="A room.")},
        starting_location=Location(canonical_name="room", name="room", desc="A room."),
        plot_considerations="",
        plot_story="Test",
        next_choices={},
    )
    prompt = orch._system_prompt(player, narrator, scene)
    assert "fortune_random(distrib='normal')" in prompt
    assert "Do not assume any character is a \"player\"" in prompt


def test_character_prompt_has_anonymity() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    char = _make_char("Alice", mock_db)
    scene = Scene(
        id="prompt_scene",
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool={char},
        starting_characters={char},
        player=char,
        narrator=char,
        location_pool=set(),
        starting_location=Location(canonical_name="room", name="room", desc="A room."),
        plot_considerations="",
        plot_story="Test",
        next_choices={},
    )
    prompt = _character_system_prompt(char, scene)
    assert "Do not assume any character is a \"player\"" in prompt
