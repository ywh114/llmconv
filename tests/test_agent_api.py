"""Integration test for the agent API."""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ara.agent.client import AgentClient
from ara.agent.server import AgentServer
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.llm.models import GameRole, StreamResult
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.persistence.save import SaveManager
from ara.world.character import Character
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.character import Importance
from ara.world.scene import Scene, Location, SceneChoice
from ara.world.story import Story, StoryStep


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
        name: str | None = None,
    ) -> StreamResult:
        self.calls.append({"role": role, "tools": tools, "tool_choice": tool_choice})
        result = self.responses[self._index]
        self._index += 1
        if print_stream:
            print(result.content, end="")
        return result

    def complete_subagent(
        self, task: str, context: str, system_prompt: str = "", max_tokens: int = 512
    ) -> str:
        return f"[sub-agent summary for: {task}]"


def _make_scene() -> Scene:
    """Build a minimal scene programmatically for testing."""
    from ara.memory.knowledge import CharacterMemory, Scratchpad
    import uuid

    mock_db = MagicMock(spec=ChromaStore)

    def make_char(name: str) -> Character:
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

    player = make_char("Player")
    narrator = make_char("Narrator")
    npc = make_char("NPC")

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
        next_choices={"end": SceneChoice(id="end", desc="The end.")},
    )


from tests.settings import TEST_SETTINGS

def _make_story(mock_client: MockLLMClient) -> tuple[Story, Scene]:
    """Build a Story wired to a mock client and a patched first scene loader."""
    db = MagicMock(spec=ChromaStore)
    story = Story(TEST_SETTINGS, db, mock_client, Path("dummy.toml"))
    test_scene = _make_scene()
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


@pytest.fixture
def agent_server():
    """Start an AgentServer in a background thread with a mocked story."""
    responses = [
        StreamResult(content="The room is quiet."),  # narrator turn
        StreamResult(content=""),                     # scratch update
    ]
    mock_client = MockLLMClient(responses)
    story, scene = _make_story(mock_client)
    decisions = [
        TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=["Say hello"],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene=None,
        ),
        TurnDecision(
            next_char=scene.narrator,
            directive="Describe the room",
            suggestions=[],
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
            next_scene="end",
        ),
    ]
    decision_iter = iter(decisions)

    def mock_decide(*args: Any, **kwargs: Any) -> TurnDecision:
        return next(decision_iter)

    story.engine.orchestrator.decide_next_turn = mock_decide  # type: ignore[method-assign]

    socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_test.sock")
    os.makedirs(TEST_SETTINGS.sockets_path, exist_ok=True)
    server = AgentServer(story, socket_path=socket_path)
    thread = threading.Thread(target=server.start_listening, daemon=True)
    thread.start()
    # Wait for the socket file to actually appear (up to 2s).
    for _ in range(40):
        if os.path.exists(socket_path):
            break
        time.sleep(0.05)

    yield server

    server.shutdown()
    import shutil
    shutil.rmtree(TEST_SETTINGS.data_dir, ignore_errors=True)


class TestAgentAPI:
    def test_client_step_limits_queue(self) -> None:
        """With client_step=1 the worker pauses after each event."""
        responses = [
            StreamResult(content="The room is quiet."),
            StreamResult(content=""),
        ]
        mock_client = MockLLMClient(responses)
        story, scene = _make_story(mock_client)
        decisions = [
            TurnDecision(
                next_char=scene.player,
                directive="",
                suggestions=["Say hello"],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene.narrator,
                directive="Describe the room",
                suggestions=[],
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
                next_scene="end",
            ),
        ]
        decision_iter = iter(decisions)

        def mock_decide(*args: Any, **kwargs: Any) -> TurnDecision:
            return next(decision_iter)

        story.engine.orchestrator.decide_next_turn = mock_decide  # type: ignore[method-assign]

        socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_step_test.sock")
        os.makedirs(TEST_SETTINGS.sockets_path, exist_ok=True)
        server = AgentServer(story, socket_path=socket_path, client_step=1)
        thread = threading.Thread(target=server.start_listening, daemon=True)
        thread.start()
        for _ in range(40):
            if os.path.exists(socket_path):
                break
            time.sleep(0.05)

        try:
            with AgentClient(socket_path) as client:
                client.start()

                # Worker should have produced exactly 1 event (scene_loaded)
                # and then paused.
                step = client.step()
                assert step["event"] == "scene_loaded"

                # After popping, worker wakes and produces the next event,
                # then pauses again.
                step = client.step()
                assert step["event"] == "needs_player_input"

                client.input("Hello there")

                step = client.step()
                assert step["event"] == "turn"

                step = client.step()
                assert step["event"] == "transition"
                assert step["phase"] == "ended"

                step = client.step()
                assert step["event"] == "story_complete"
        finally:
            server.shutdown()
            import shutil
            shutil.rmtree(TEST_SETTINGS.data_dir, ignore_errors=True)

    def test_full_session(self, agent_server: AgentServer) -> None:
        socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_test.sock")
        with AgentClient(socket_path) as client:
            # 1. Start the story
            result = client.start()
            assert result["finished"] is False
            assert result["scene_history"] == []

            # 2. First step loads the scene
            step = client.step()
            assert step["event"] == "scene_loaded"
            assert step["scene"]["id"] == "test"
            assert step["scene"]["player"] == "Player"

            # 3. Next step requests player input
            step = client.step()
            assert step["event"] == "needs_player_input"
            assert step["suggestions"] == ["Say hello"]

            # 4. Submit player input
            inp = client.input("Hello there")
            assert inp["submitted"] == "Hello there"

            # 5. Narrator turn
            step = client.step()
            assert step["event"] == "turn"
            assert "The room is quiet" in step["output"]

            # 6. Get full state snapshot
            state = client.state()
            assert state["engine"]["location"] == "room"
            assert state["story"]["current_scene"]["id"] == "test"
            assert state["engine"]["last_decision"]["next_char"] == "Narrator"

            # 7. Scene ends
            step = client.step()
            assert step["event"] == "transition"
            assert step["phase"] == "ended"
            assert step["next_scene"] == "end"

            # 8. Next step finalises and tries to load the follow-up scene, which does not exist
            step = client.step()
            assert step["event"] == "story_complete"

    def test_skip(self, agent_server: AgentServer) -> None:
        socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_test.sock")
        with AgentClient(socket_path) as client:
            client.start()
            client.step()  # scene_loaded

            # Skip to a non-existent scene should raise an error
            with pytest.raises(RuntimeError, match="not found"):
                client.skip("nonexistent")

    def test_run_until_input(self, agent_server: AgentServer) -> None:
        socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_test.sock")
        with AgentClient(socket_path) as client:
            client.start()
            client.step()  # scene_loaded

            # run_until_input should advance straight to the player prompt
            result = client.run_until_input()
            assert len(result["events"]) == 1
            assert result["events"][0]["event"] == "needs_player_input"

    def test_debug_commands(self, agent_server: AgentServer) -> None:
        socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_test.sock")
        with AgentClient(socket_path) as client:
            client.start()
            client.step()  # scene_loaded
            client.step()  # needs_player_input

            # debug: info
            info = client.debug("info")
            assert info["scene"] == "test"
            assert info["location"] == "room"
            here_names = {c["name"] for c in info["here"]}
            assert here_names == {"Player", "Narrator", "NPC"}

            # debug: here
            here = client.debug("here")
            names = {c["name"] for c in here["characters"]}
            assert names == {"Player", "Narrator", "NPC"}

            # debug: away
            away = client.debug("away")
            assert away["characters"] == []

            # debug: loc
            loc = client.debug("loc")
            assert loc["name"] == "room"

            # debug: scene
            sc = client.debug("scene")
            assert sc["id"] == "test"
            assert sc["language"] == "English"

            # debug: scratch
            scratch = client.debug("scratch", args=["NPC"])
            assert scratch["character"] == "NPC"
            assert "scratch" in scratch

            # debug: summary
            summary = client.debug("summary", args=["NPC"])
            assert summary["character"] == "NPC"
            assert "prev_scene_summary" in summary

            # debug: decision (should be the player-input decision)
            dec = client.debug("decision")
            assert dec["next"] == "Player"
            assert dec["suggestions"] == ["Say hello"]

            # debug: help
            help_resp = client.debug("help")
            assert "commands" in help_resp

            # debug: dump
            dump = client.debug("dump")
            assert "messages" in dump
            # Context may be empty before any NPC/narrator turns



def _make_scene_with_enter() -> Scene:
    """Build a scene where one character starts away."""
    mock_db = MagicMock(spec=ChromaStore)

    def make_char(name: str) -> Character:
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

    player = make_char("Player")
    narrator = make_char("Narrator")
    npc = make_char("NPC")

    loc = Location(canonical_name="room", name="room", desc="A room.")

    return Scene(
        id="test",
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool={player, narrator, npc},
        starting_characters={player, narrator},  # NPC starts AWAY
        player=player,
        narrator=narrator,
        location_pool={loc},
        starting_location=loc,
        plot_considerations="",
        plot_story="Test scene",
        next_choices={"end": SceneChoice(id="end", desc="The end.")},
    )


def _make_story_with_enter(mock_client: MockLLMClient) -> tuple[Story, Scene]:
    """Build a Story where NPC starts away."""
    db = MagicMock(spec=ChromaStore)
    story = Story(TEST_SETTINGS, db, mock_client, Path("test/dummy.toml"))
    test_scene = _make_scene_with_enter()

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


@pytest.fixture
def agent_server_enter():
    """AgentServer with a scene where NPC starts away."""
    responses = [
        StreamResult(content="The room is quiet."),
        StreamResult(content=""),
    ]
    mock_client = MockLLMClient(responses)
    story, scene = _make_story_with_enter(mock_client)

    # Find NPC character
    npc_char = next(c for c in scene.character_pool if c.name == "NPC")

    # First turn: orchestrator decides NPC enters, then narrator speaks
    decisions = [
        TurnDecision(
            next_char=scene.narrator,
            directive="Describe the room",
            suggestions=[],
            entering_chars={npc_char},
            exiting_chars=set(),
            switch_location=None,
            next_scene=None,
        ),
        TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=["Say hello"],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene="end",
        ),
    ]
    decision_iter = iter(decisions)

    def mock_decide(*args, **kwargs):
        try:
            return next(decision_iter)
        except StopIteration:
            # Gracefully end the scene when decisions run out.
            return TurnDecision(
                next_char=scene.narrator,
                directive="",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                next_scene="end",
            )

    story.engine.orchestrator.decide_next_turn = mock_decide  # type: ignore[method-assign]

    socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_enter_test.sock")
    os.makedirs(TEST_SETTINGS.sockets_path, exist_ok=True)
    server = AgentServer(story, socket_path=socket_path)
    thread = threading.Thread(target=server.start_listening, daemon=True)
    thread.start()
    for _ in range(40):
        if os.path.exists(socket_path):
            break
        time.sleep(0.05)

    yield server

    server.shutdown()
    shutil.rmtree(TEST_SETTINGS.data_dir, ignore_errors=True)


class TestSaveLoadEnterExit:
    def test_save_load_preserves_here_away(self, agent_server_enter: AgentServer) -> None:
        """After an enter, save and reload; NPC must still be in here_chars."""
        socket_path = str(TEST_SETTINGS.sockets_path / "ara_agent_enter_test.sock")
        with AgentClient(socket_path) as client:
            # 1. Start
            client.start()

            # 2. Load scene
            step = client.step()
            assert step["event"] == "scene_loaded"

            # 3. Narrator turn (NPC enters during this turn)
            step = client.step()
            assert step["event"] == "turn"
            assert step["speaker"] == "Narrator"
            assert "NPC" in step["enter"]

            # 4. Check state: NPC should now be here
            state = client.state()
            here = {c["name"] for c in state["engine"]["here"]}
            away = {c["name"] for c in state["engine"]["away"]}
            assert "NPC" in here
            assert "NPC" not in away

            # 5. Save
            save_result = client.save(slot=99)
            assert save_result["slot"] == 99

            # 6. Reset (clear state)
            client.reset()

            # 7. Load
            client.load(slot=99)

            # 8. Check state again
            state = client.state()
            here = {c["name"] for c in state["engine"]["here"]}
            away = {c["name"] for c in state["engine"]["away"]}
            assert "NPC" in here, f"NPC should be in here after load, got: {here}"
            assert "NPC" not in away, f"NPC should not be in away after load, got: {away}"

            # 9. Clean up save file
            manager = SaveManager(TEST_SETTINGS)
            manager.delete(agent_server_enter.story._story_dir.name, 99)
