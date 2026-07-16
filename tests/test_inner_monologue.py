"""Tests for inner monologue response modes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.models import StreamResult
from ara.memory.chroma import ChromaStore
from ara.persistence.save import SaveManager
from ara.world.engine import Engine, _parse_inner_response
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Scene
from ara.world.story import Story

from tests.helpers import make_scene


def _make_scene(mock_db: ChromaStore) -> Scene:
    return make_scene(
        "inner_scene", mock_db, char_names=("Player", "Narrator", "Alice")
    )


def test_parse_inner_response_json() -> None:
    raw = json.dumps({"outer": "Hello", "inner": "I am scared", "explain": "fear"})
    outer, inner, explain = _parse_inner_response(raw)
    assert outer == "Hello"
    assert inner == "I am scared"
    assert explain == "fear"


def test_parse_inner_response_fallback() -> None:
    raw = "Just plain text."
    outer, inner, explain = _parse_inner_response(raw)
    assert outer == "Just plain text."
    assert inner is None
    assert explain is None


def test_character_outer_and_inner_stores_private_thought() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene(mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")

    engine = Engine(mock_client, db=None)
    engine.start(scene)

    decision = TurnDecision(
        next_char=alice,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        response_mode="outer_and_inner",
    )
    mock_client.complete.return_value = StreamResult(
        content=json.dumps({
            "outer": "Nice weather today.",
            "inner": "I hope nobody notices the knife.",
            "explain": "hiding intent",
        }),
        tool_calls=[],
        reasoning_content="",
    )

    output = engine._character_turn(scene, engine.ctx, decision, engine.loc)
    assert output == "Nice weather today."
    assert len(alice.inner_log) == 1
    assert alice.inner_log[0]["inner"] == "I hope nobody notices the knife."

    # Shared context should only contain the outer line.
    public_messages = [msg for msg in engine.ctx.context if msg.get("name") == alice.name]
    assert len(public_messages) == 1
    assert public_messages[0]["content"] == "Nice weather today."


def test_character_inner_only_silent_turn() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene(mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")

    engine = Engine(mock_client, db=None)
    engine.start(scene)

    decision = TurnDecision(
        next_char=alice,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        response_mode="inner_only",
    )
    mock_client.complete.return_value = StreamResult(
        content=json.dumps({
            "outer": "",
            "inner": "They must not find me.",
            "explain": "stealth",
        }),
        tool_calls=[],
        reasoning_content="",
    )

    output = engine._character_turn(scene, engine.ctx, decision, engine.loc)
    assert output == ""
    assert len(alice.inner_log) == 1
    assert alice.inner_log[0]["inner"] == "They must not find me."
    # No public assistant message should be added for a silent turn.
    public_messages = [msg for msg in engine.ctx.context if msg.get("name") == alice.name]
    assert not public_messages


def test_save_load_preserves_inner_log() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene(mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")
    alice.inner_log.append({"outer": "Hi", "inner": "secret", "explain": "test"})

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "inner_scene.toml").write_text('id = "inner_scene"\nname = "Inner Scene"\n')

        story = Story(config, mock_db, mock_client, tmp / "inner_scene.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene

        manager = SaveManager(config)
        path = manager.save(story, slot=1)
        snapshot = json.loads(path.read_text())
        alice_data = next(c for c in snapshot["characters"] if c["name"] == "Alice")
        assert alice_data["inner_log"]

        fresh_story = Story(config, mock_db, mock_client, tmp / "inner_scene.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        fresh_alice = next(c for c in fresh_story.current_scene.character_pool if c.name == "Alice")
        assert fresh_alice.inner_log == alice.inner_log
