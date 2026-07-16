"""Tests for the plot time= field and runtime time changes."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from ara.world.character import Character, Importance
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene


def _make_char(name: str) -> Character:
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
        memory=MagicMock(),
        scratch=MagicMock(),
    )


def _make_scene() -> Scene:
    player = _make_char("Player")
    narrator = _make_char("Narrator")
    npc = _make_char("NPC")
    loc = Location(canonical_name="room", name="room", desc="A room.")
    return Scene(
        id="test",
        language="English",
        zeitgeist="test",
        tone="neutral",
        time="morning",
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


def test_scene_loads_time() -> None:
    """A scene with time='morning' exposes it through state."""
    scene = _make_scene()
    assert scene.time == "morning"


def test_engine_world_time_from_scene() -> None:
    """Engine.world_time is initialised from scene.time."""
    scene = _make_scene()

    class MockClient:
        def complete(self, **kwargs):
            from ara.llm.models import StreamResult
            return StreamResult(content="mock")

    engine = Engine(MockClient())  # type: ignore[arg-type]
    engine.start(scene)
    assert engine.world_time == "morning"


def test_orchestrator_set_time() -> None:
    """The orchestrator can change world time via set_time."""
    scene = _make_scene()

    class MockClient:
        def complete(self, **kwargs):
            from ara.llm.models import StreamResult
            return StreamResult(content="mock")

    engine = Engine(MockClient())  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.return_value = TurnDecision(
        next_char=scene.narrator,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        set_time="night",
    )
    engine.start(scene)
    engine.step()
    assert engine.world_time == "night"
