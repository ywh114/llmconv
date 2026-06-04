"""Smoke-test the CLI state-machine loop with a mocked LLM."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.models import StreamResult
from ara.world.character import Character
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene
from ara.world.story import Story


def _make_char(name: str, mock_db) -> Character:
    cid = uuid.uuid5(uuid.NAMESPACE_DNS, f"test.{name}")
    return Character(
        id=cid,
        name=name,
        card_fields={
            "name": name,
            "summary": f"{name} summary",
            "personality": f"{name} personality",
            "scenario": f"{name} scenario",
            "greeting_message": f"Hi, I'm {name}",
            "example_messages": "",
        },
        importance=1,
        memory=CharacterMemory(character_id=cid, db=mock_db),
        scratch=Scratchpad(),
    )


def _make_scene(mock_db) -> Scene:
    player = _make_char("Player", mock_db)
    narrator = _make_char("Narrator", mock_db)
    npc = _make_char("NPC", mock_db)
    loc = Location(name="room", desc="A room.")
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


def test_state_machine_story_with_next_choice() -> None:
    """Drive a story with an actual next-scene choice."""
    mock_db = MagicMock(spec=ChromaStore)
    scene = _make_scene(mock_db)
    scene.next_choices = {"tea": MagicMock(id="tea")}

    engine_decisions = [
        TurnDecision(
            next_char=scene.narrator,
            directive="Introduce",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene=None,
        ),
        TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=["Go to tea"],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene=None,
        ),
        TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene="tea",
        ),
    ]

    class MockClient:
        def complete(self, **kwargs):
            return StreamResult(content="mock")

        def complete_subagent(self, **kwargs):
            return "[sub-agent summary]"

    mock = MockClient()
    engine = Engine(mock)  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.side_effect = lambda **kw: engine_decisions.pop(0)

    story = Story(AraSettings(), mock_db, mock, Path(__file__))
    story.engine = engine

    with patch("ara.world.story.Scene.load", return_value=scene):
        story.start()
        assert not story.finished
        assert story._state == "loading"

        # Tick 1: load scene
        result = story.step()
        assert result.event == "scene_loaded", f"expected scene_loaded, got {result.event}"
        assert result.scene is scene
        print(f"[OK] scene_loaded: {result.scene.id}")

        # Tick 2: narrator turn
        result = story.step()
        assert result.event == "turn", f"expected turn, got {result.event}"
        assert engine.last_decision.next_char == scene.narrator
        print("[OK] narrator turn")

        # Tick 3: player turn requested
        result = story.step()
        assert result.event == "needs_player_input", f"expected needs_player_input, got {result.event}"
        assert result.suggestions == ["Go to tea"]
        print(f"[OK] needs_player_input: suggestions={result.suggestions}")

        # Provide player input
        story.submit_player_input("Let's go")
        assert not engine.needs_player_input
        print("[OK] player input submitted")

        # Tick 4: scene ends
        result = story.step()
        assert result.event == "scene_ended", f"expected scene_ended, got {result.event}"
        assert result.next_scene == "tea"
        print(f"[OK] scene_ended -> {result.next_scene}")

        # Next tick tries to load "tea.toml" which doesn't exist -> story_complete
        result = story.step()
        assert result.event == "story_complete", f"expected story_complete, got {result.event}"
        print("[OK] story_complete (next scene file missing)")

        assert story.finished
        print("\nAll state-machine smoke tests passed!")


if __name__ == "__main__":
    test_state_machine_story_with_next_choice()
