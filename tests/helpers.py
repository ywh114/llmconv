"""Shared factories and fakes for Ara unit tests.

These helpers replace the ``_make_char`` / ``_make_scene`` / ``MockLLMClient``
copies that used to be duplicated across test modules.  Import from here
instead of redefining them locally or importing them from another test file.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ara.llm.models import GameRole, StreamResult
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.world.character import Character, Importance
from ara.world.item import Item
from ara.world.scene import Location, Scene, SceneChoice


def stable_cid(name: str) -> uuid.UUID:
    """Return the deterministic character id used throughout the test suite."""
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"test.{name}")


def make_char(
    name: str,
    mock_db: ChromaStore,
    *,
    importance: Importance = Importance.IMPORTANT,
) -> Character:
    """Build a Character backed by (mocked) vector memory and a scratchpad."""
    cid = stable_cid(name)
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
        importance=importance,
        memory=CharacterMemory(character_id=cid, db=mock_db),
        scratch=Scratchpad(),
    )


def make_scene(
    scene_id: str = "test",
    mock_db: ChromaStore | None = None,
    *,
    char_names: tuple[str, ...] = ("Player", "Narrator", "NPC"),
    next_choices: dict[str, SceneChoice] | None = None,
    location_name: str = "room",
    items: dict[str, Item] | None = None,
) -> Scene:
    """Build a minimal scene.  ``char_names`` must include Player and Narrator."""
    chars = {make_char(name, mock_db) for name in char_names}
    player = next(c for c in chars if c.name == "Player")
    narrator = next(c for c in chars if c.name == "Narrator")
    loc = Location(
        canonical_name=location_name,
        name=location_name,
        desc=f"A {location_name}.",
    )
    return Scene(
        id=scene_id,
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
        plot_story=f"Test {scene_id}",
        next_choices=next_choices or {},
        items=items or {},
    )


def make_scene_with_chars(chars: list[Character]) -> Scene:
    """Build a minimal scene from explicit characters.

    The character named ``Narrator`` narrates when present; otherwise the
    Player doubles as narrator (mirrors the historical test helper).
    """
    player = next(c for c in chars if c.name == "Player")
    narrator = next((c for c in chars if c.name == "Narrator"), player)
    loc = Location(canonical_name="room", name="room", desc="A room.")
    return Scene(
        id="test",
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool=set(chars),
        starting_characters=set(chars),
        player=player,
        narrator=narrator,
        location_pool={loc},
        starting_location=loc,
        plot_considerations="",
        plot_story="Test scene",
        next_choices={},
    )


def make_next_round_result(next_char: str, response_mode: str = "outer") -> StreamResult:
    """Build an LLM result carrying a ``next_round`` tool call for *next_char*."""
    return StreamResult(
        content="",
        tool_calls=[{
            "id": "call_next",
            "type": "function",
            "function": {
                "name": "next_round",
                "arguments": json.dumps({
                    "next_character": next_char,
                    "directive": "",
                    "suggestions": [],
                    "enter_characters": [],
                    "exit_characters": [],
                    "switch_location": "",
                    "edit_location": "",
                    "end_scene": False,
                    "next_scene": "",
                    "response_mode": response_mode,
                }),
            },
        }],
    )


class ScriptedLLMClient:
    """Fake LLM client that returns pre-canned responses in order."""

    def __init__(
        self,
        responses: list[StreamResult],
        subagent_answer: str | None = None,
    ) -> None:
        self.responses = responses
        self.subagent_answer = subagent_answer
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        role: GameRole,
        system_prompt: str = "",
        messages: list | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        stream: bool = True,
        print_stream: bool = False,
        name: str | None = None,
    ) -> StreamResult:
        self.calls.append({
            "role": role,
            "system_prompt": system_prompt,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        })
        result = self.responses[self._index]
        self._index += 1
        if print_stream:
            print(result.content, end="")
        return result

    def complete_subagent(
        self, task: str, context: str, system_prompt: str = "", max_tokens: int = 512
    ) -> str:
        if self.subagent_answer is not None:
            return self.subagent_answer
        return f"[sub-agent summary for: {task}]"
