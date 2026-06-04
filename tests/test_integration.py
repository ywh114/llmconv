"""Integration tests using a mocked LLM client."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.models import GameRole, StreamResult
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Scene


class MockLLMClient:
    """Fake LLM client that returns pre-canned responses."""

    def __init__(self, responses: list[StreamResult]) -> None:
        self.responses = responses
        self._index = 0
        self.calls: list[dict] = []

    def complete(
        self,
        role: GameRole,
        system_prompt: str,
        messages: list,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        stream: bool = True,
        print_stream: bool = False,
    ) -> StreamResult:
        self.calls.append({
            "role": role,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        result = self.responses[self._index]
        self._index += 1
        return result

    def complete_subagent(
        self, task: str, context: str, system_prompt: str = "", max_tokens: int = 512
    ) -> str:
        return f"[sub-agent summary for: {task}]"


def _make_scene() -> Scene:
    """Build a minimal scene programmatically for testing."""
    from ara.memory.knowledge import CharacterMemory, Scratchpad
    from ara.world.character import Character
    import uuid

    mock_db = MagicMock(spec=ChromaStore)

    def make_char(name: str) -> Character:
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

    player = make_char("Player")
    narrator = make_char("Narrator")
    npc = make_char("NPC")

    from ara.world.scene import Location
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


class TestEngineLoop:
    """End-to-end test of the conversation engine with a mock LLM."""

    def test_scene_ends_on_next_scene(self) -> None:
        """The engine should exit when the orchestrator returns a next_scene."""
        scene = _make_scene()

        # Orchestrator decides to end scene immediately
        orchestrator_result = StreamResult(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "next_round",
                        "arguments": json.dumps({
                            "next_character": "Player",
                            "directive": "",
                            "suggestions": [],
                            "enter_characters": [],
                            "exit_characters": [],
                            "switch_location": "",
                            "next_scene": "end",
                        }),
                    },
                }
            ],
        )

        mock = MockLLMClient([orchestrator_result])
        engine = Engine(mock)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene="end",
        )

        inputs = ["hello"]
        def fake_input(prompt: str, suggestions: list[str]) -> str:
            return inputs.pop(0)

        result = engine.run(scene, get_user_input=fake_input)
        assert result == "end"

    def test_edit_location_updates_description(self) -> None:
        """The engine should update the location description when edit_location is set."""
        scene = _make_scene()
        original_desc = scene.starting_location.desc

        mock = MockLLMClient([StreamResult(content="The room is quiet.")])
        engine = Engine(mock)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=scene.narrator,
            directive="Describe the broken table",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            edit_location="The table was shattered into splinters.",
            next_scene=None,
        )

        engine.start(scene)
        engine.step()

        assert "shattered into splinters" in scene.starting_location.desc
        assert original_desc in scene.starting_location.desc
