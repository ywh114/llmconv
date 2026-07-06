"""Tests for character-scoped wiki_recall tool."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

from ara.config import AraSettings
from ara.llm.models import StreamResult
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.world.character import Character, Importance
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene


def _make_char(name: str, importance: Importance, mock_db: ChromaStore) -> Character:
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
        importance=importance,
        memory=CharacterMemory(character_id=cid, db=mock_db),
        scratch=Scratchpad(),
    )


def _make_scene_with_chars(chars: list[Character]) -> Scene:
    player = next(c for c in chars if c.name == "Player")
    narrator = next(c for c in chars if c.name == "Player")
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


def _next_round_result(next_char: str) -> StreamResult:
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
                    "response_mode": "outer",
                }),
            },
        }],
    )


def test_character_wiki_recall_tool_exists() -> None:
    """IMPORTANT characters should receive a wiki_recall tool."""
    mock_db = MagicMock(spec=ChromaStore)
    npc = _make_char("NPC", Importance.IMPORTANT, mock_db)
    player = _make_char("Player", Importance.EIGEN, mock_db)
    scene = _make_scene_with_chars([player, npc])

    calls: list[dict] = []

    class _FakeClient:
        def complete(self, **kwargs):
            calls.append(kwargs)
            return StreamResult(content="I looked it up.")

        def complete_subagent(self, **kwargs):
            return ""

    engine = Engine(_FakeClient(), db=mock_db)  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.return_value = TurnDecision(
        next_char=npc,
        directive="Recall something",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        edit_location="",
        next_scene=None,
    )

    engine.start(scene)
    engine.step()

    assert calls[-1]["tools"] is not None
    tool_names = {t["function"]["name"] for t in calls[-1]["tools"]}
    assert "wiki_recall" in tool_names


def test_character_wiki_recall_uses_querier_filter() -> None:
    """Calling wiki_recall from a character turn passes the character as querier."""
    mock_db = MagicMock(spec=ChromaStore)
    npc = _make_char("NPC", Importance.IMPORTANT, mock_db)
    player = _make_char("Player", Importance.EIGEN, mock_db)
    scene = _make_scene_with_chars([player, npc])

    class _FakeClient:
        def __init__(self):
            self._calls = 0

        def complete(self, **kwargs):
            self._calls += 1
            if self._calls == 1:
                return StreamResult(
                    content="",
                    tool_calls=[{
                        "id": "call_wiki",
                        "type": "function",
                        "function": {
                            "name": "wiki_recall",
                            "arguments": json.dumps({"query": "major sects"}),
                        },
                    }],
                )
            return StreamResult(content="I know about the sects now.")

        def complete_subagent(self, **kwargs):
            return ""

    fake_client = _FakeClient()
    engine = Engine(fake_client, db=mock_db)  # type: ignore[arg-type]
    engine.orchestrator._wiki_recall = MagicMock(return_value="Qingyun and Heavenly Sword.")
    engine.orchestrator.decide_next_turn = MagicMock(return_value=TurnDecision(
        next_char=npc,
        directive="Recall something",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        edit_location="",
        next_scene=None,
    ))

    engine.start(scene)
    engine.step()

    engine.orchestrator._wiki_recall.assert_called_once()
    call_args = engine.orchestrator._wiki_recall.call_args
    assert call_args.kwargs.get("querier") is npc
    assert call_args.args[0] == "major sects"
