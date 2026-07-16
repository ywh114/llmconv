"""Tests for character-scoped wiki_recall tool."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from ara.llm.models import StreamResult
from ara.memory.chroma import ChromaStore
from ara.world.character import Character, Importance
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision

from tests.helpers import make_char as _make_char_impl
from tests.helpers import make_scene_with_chars as _make_scene_with_chars


def _make_char(name: str, importance: Importance, mock_db: ChromaStore) -> Character:
    return _make_char_impl(name, mock_db, importance=importance)


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
    engine.orchestrator.wiki.recall = MagicMock(return_value="Qingyun and Heavenly Sword.")
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

    engine.orchestrator.wiki.recall.assert_called_once()
    call_args = engine.orchestrator.wiki.recall.call_args
    assert call_args.kwargs.get("querier") is npc
    assert call_args.args[0] == "major sects"
