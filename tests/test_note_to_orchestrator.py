"""Tests for the attempt_action tool."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ara.agent.client import AgentClient
from ara.agent.server import AgentServer
from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.models import GameRole, StreamResult
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.persistence.save import SaveManager
from ara.world.character import Character, Importance
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene, SceneChoice
from ara.world.story import Story, StoryStep

from tests.settings import TEST_SETTINGS


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
    chars = {_make_char(name, mock_db) for name in ["Player", "Narrator", "Alice"]}
    player = next(c for c in chars if c.name == "Player")
    narrator = next(c for c in chars if c.name == "Narrator")
    loc = Location(canonical_name="room", name="room", desc="A room.")
    return Scene(
        id="note_scene",
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
        plot_story="Test note tool",
        next_choices={},
    )


def test_character_attempt_action_stores_attempt() -> None:
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
    )

    # First LLM call returns the tool call; second call returns spoken text.
    mock_client.complete.side_effect = [
        StreamResult(
            content="",
            tool_calls=[{
                "id": "call_note",
                "type": "function",
                "function": {
                    "name": "attempt_action",
                    "arguments": json.dumps({
                        "action": "I try to stab the goblin",
                        "intent": "kill it quickly",
                        "target": "goblin",
                        "secrecy": "loud",
                    }),
                },
            }],
        ),
        StreamResult(content="Take that!", tool_calls=[], reasoning_content=""),
    ]

    output = engine._character_turn(scene, engine.ctx, decision, engine.loc)
    assert output == "Take that!"
    assert len(engine._pending_attempts) == 1
    attempt = engine._pending_attempts[0]
    assert attempt["source"] == "Alice"
    assert attempt["action"] == "I try to stab the goblin"
    assert attempt["target"] == "goblin"


def test_engine_passes_attempts_to_orchestrator_and_clears() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene(mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")

    engine = Engine(mock_client, db=None)
    engine.start(scene)
    engine._pending_attempts.append({
        "source": "Alice",
        "action": "I try to stab the goblin",
    })

    original_decide = engine.orchestrator.decide_next_turn
    captured_attempts: list[dict] | None = None

    def mock_decide(*args, **kwargs):
        nonlocal captured_attempts
        captured_attempts = kwargs.get("attempts_for_orchestrator")
        return TurnDecision(
            next_char=alice,
            directive="",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
        )

    engine.orchestrator.decide_next_turn = mock_decide  # type: ignore[method-assign]
    mock_client.complete.return_value = StreamResult(
        content="Alice speaks.", tool_calls=[], reasoning_content=""
    )

    engine.step()

    assert captured_attempts is not None
    assert len(captured_attempts) == 1
    assert captured_attempts[0]["action"] == "I try to stab the goblin"
    assert engine._pending_attempts == []


def _make_story(mock_client: MagicMock) -> tuple[Story, Scene]:
    db = MagicMock(spec=ChromaStore)
    story = Story(TEST_SETTINGS, db, mock_client, Path("test/dummy.toml"))
    test_scene = _make_scene(db)
    original_load = story._load_scene

    def _patched_load() -> StoryStep:
        story._load_scene = original_load
        story._current_scene = test_scene
        story._scene_history.append(test_scene.id)
        story._state = "running"
        story.engine.start(test_scene)
        return StoryStep(event="scene_loaded", scene=test_scene)

    story._load_scene = _patched_load  # type: ignore[method-assign]
    return story, test_scene


def test_agent_api_attempt_then_reply() -> None:
    responses = [
        StreamResult(content="The room is quiet."),
        StreamResult(content=""),
    ]
    mock_client = MagicMock(spec=LLMClient)
    mock_client.complete.side_effect = responses
    story, scene = _make_story(mock_client)

    decisions = [
        TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=["Say hello"],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
        ),
        TurnDecision(
            next_char=scene.narrator,
            directive="Describe the room",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
        ),
        TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene="end",
        ),
    ]
    decision_iter = iter(decisions)

    def mock_decide(*args, **kwargs):
        return next(decision_iter)

    story.engine.orchestrator.decide_next_turn = mock_decide  # type: ignore[method-assign]

    import os
    import shutil
    import threading
    import time

    socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_note_test.sock")
    os.makedirs(TEST_SETTINGS.sockets_path, exist_ok=True)
    server = AgentServer(story, socket_path=socket_path)
    thread = threading.Thread(target=server.start_listening, daemon=True)
    thread.start()
    for _ in range(40):
        if os.path.exists(socket_path):
            break
        time.sleep(0.05)

    try:
        with AgentClient(socket_path) as client:
            client.start()
            client.step()  # scene_loaded
            client.step()  # needs_player_input

            # Store a note without ending the turn.
            attempt_result = client.attempt("I try to stab the goblin")
            assert attempt_result["attempted"] == "I try to stab the goblin"
            assert story.engine._pending_attempts

            # Now reply; the attempt should be bundled.
            input_result = client.input("waaaghhhh")
            assert input_result["submitted"] == "waaaghhhh"

            # After step consumes the decision, pending attempts should be cleared.
            # We verify by stepping and checking no attempts remain.
            client.step()
            assert story.engine._pending_attempts == []
    finally:
        server.shutdown()
        shutil.rmtree(TEST_SETTINGS.data_dir, ignore_errors=True)


def test_save_load_preserves_pending_attempts() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene(mock_db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "note_scene.toml").write_text('id = "note_scene"\nname = "Note Scene"\n')

        story = Story(config, mock_db, mock_client, tmp / "note_scene.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene
        story.engine.submit_attempt("I try to sneak past", source="Player")

        manager = SaveManager(config)
        path = manager.save(story, slot=1)
        snapshot = json.loads(path.read_text())
        assert snapshot["engine_state"]["_pending_attempts"]

        fresh_story = Story(config, mock_db, mock_client, tmp / "note_scene.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        assert fresh_story.engine._pending_attempts
        assert fresh_story.engine._pending_attempts[0]["action"] == "I try to sneak past"
