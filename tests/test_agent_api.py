"""Integration test for the agent API."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ara.agent.client import AgentClient
from ara.agent.server import AgentServer
from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.models import GameRole, StreamResult
from ara.world.character import Character
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.models import Importance
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
        next_choices={"end": SceneChoice(id="end", desc="The end.")},
    )


def _make_story(mock_client: MockLLMClient) -> tuple[Story, Scene]:
    """Build a Story wired to a mock client and a patched first scene loader."""
    settings = AraSettings()
    db = MagicMock(spec=ChromaStore)
    story = Story(settings, db, mock_client, Path("dummy.toml"))
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

    socket_path = "sockets/ara_agent_test.sock"
    os.makedirs("sockets", exist_ok=True)
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


class TestAgentAPI:
    def test_full_session(self, agent_server: AgentServer) -> None:
        with AgentClient("sockets/ara_agent_test.sock") as client:
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
            assert state["engine"]["needs_player_input"] is False
            assert state["engine"]["location"] == "room"
            assert state["story"]["current_scene"]["id"] == "test"
            assert state["engine"]["last_decision"]["next_char"] == "Narrator"

            # 7. Scene ends
            step = client.step()
            assert step["event"] == "scene_ended"
            assert step["next_scene"] == "end"

            # 8. Next step tries to load the follow-up scene, which does not exist
            step = client.step()
            assert step["event"] == "story_complete"

    def test_skip(self, agent_server: AgentServer) -> None:
        with AgentClient("sockets/ara_agent_test.sock") as client:
            client.start()
            client.step()  # scene_loaded

            # Skip to a non-existent scene should raise an error
            with pytest.raises(RuntimeError, match="not found"):
                client.skip("nonexistent")

    def test_run_until_input(self, agent_server: AgentServer) -> None:
        with AgentClient("sockets/ara_agent_test.sock") as client:
            client.start()
            client.step()  # scene_loaded

            # run_until_input should advance straight to the player prompt
            result = client.run_until_input()
            assert len(result["events"]) == 1
            assert result["events"][0]["event"] == "needs_player_input"

    def test_debug_commands(self, agent_server: AgentServer) -> None:
        with AgentClient("sockets/ara_agent_test.sock") as client:
            client.start()
            client.step()  # scene_loaded
            client.step()  # needs_player_input

            # debug: info
            info = client.debug("info")
            assert info["scene"] == "test"
            assert info["location"] == "room"
            assert set(info["here"]) == {"Player", "Narrator", "NPC"}

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
