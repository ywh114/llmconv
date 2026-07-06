"""Tests for :mod:`ara.agent.state`."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from ara.agent.state import engine_to_dict
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.world.character import Character, Importance
from ara.world.engine import Engine
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


def _make_scene(mock_db: ChromaStore) -> Scene:
    player = _make_char("Player", mock_db)
    narrator = _make_char("Narrator", mock_db)
    npc = _make_char("NPC", mock_db)
    loc = Location(canonical_name="room", name="room", desc="A room.")
    return Scene(
        id="test",
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool={player, narrator, npc},
        starting_characters={player, narrator, npc},
        player=player,
        narrator=narrator,
        location_pool={loc},
        starting_location=loc,
        plot_considerations="",
        plot_story="Test scene",
        next_choices={},
    )


def test_character_statuses_have_default_title() -> None:
    """Empty character statuses should still expose the default page title."""
    mock_db = MagicMock(spec=ChromaStore)
    scene = _make_scene(mock_db)
    engine = Engine(MagicMock(), db=mock_db)  # type: ignore[arg-type]
    engine.start(scene)

    state = engine_to_dict(engine)

    assert "character_statuses" in state
    for name, status in state["character_statuses"].items():
        assert status.get("title") == "Status", f"{name} status missing default title"


def test_location_statuses_have_default_title() -> None:
    """Empty location statuses should still expose the default page title."""
    mock_db = MagicMock(spec=ChromaStore)
    scene = _make_scene(mock_db)
    engine = Engine(MagicMock(), db=mock_db)  # type: ignore[arg-type]
    engine.start(scene)

    state = engine_to_dict(engine)

    assert "location_statuses" in state
    for name, status in state["location_statuses"].items():
        assert status.get("title") == "Status", f"{name} status missing default title"
