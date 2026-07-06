"""End-to-end test of the aractl CLI against a live (mock-backed) agent server."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ara.agent.server import AgentServer
from ara.config import AraSettings
from ara.llm.models import GameRole, StreamResult
from ara.memory.chroma import ChromaStore
from ara.world.character import Character, Importance
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene, SceneChoice
from ara.world.story import Story


class MockLLMClient:
    """Fake LLM client that returns pre-canned responses."""

    def __init__(self, responses: list[StreamResult]) -> None:
        self.responses = responses
        self._index = 0
        self.calls: list[dict] = []

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
        self.calls.append({"role": role, "tools": tools, "tool_choice": tool_choice})
        result = self.responses[self._index]
        self._index += 1
        if print_stream:
            print(result.content, end="")
        return result

    def complete_subagent(
        self, task: str, context: str, system_prompt: str = "", max_tokens: int = 512
    ) -> str:
        return ""


def _stable_cid(name: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"test.{name}")


def _make_char(name: str, mock_db: ChromaStore) -> Character:
    return Character(
        id=_stable_cid(name),
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


def _write_assets(tmp: Path) -> Path:
    """Create a minimal data directory with a world, characters, location and scenes."""
    # World setting with arbitrary categories (validates ticket 12 generic categories).
    world_dir = tmp / "assets" / "world"
    world_dir.mkdir(parents=True)
    (world_dir / "test_world.toml").write_text(
        '''id = "test_world"
name = "Test World"
summary = "A world for e2e testing."

[[factions]]
name = "Test Faction"
description = "A faction for testing generic categories."

[[custom_category]]
name = "Custom Fact"
detail = "Arbitrary top-level categories work."
'''
    )

    # Characters.
    for name in ("Player", "Narrator", "NPC"):
        char_dir = tmp / "assets" / "cc" / "e2e" / name
        char_dir.mkdir(parents=True)
        (char_dir / "card.toml").write_text(
            f'''name = "{name}"
summary = "{name} summary"
personality = "{name} personality"
scenario = "{name} scenario"
greeting_message = "Hi, I'm {name}"
example_messages = ""
'''
        )
        (char_dir / "meta.toml").write_text('importance = "IMPORTANT"\n')

    # Location.
    loc_dir = tmp / "assets" / "lc" / "e2e" / "room"
    loc_dir.mkdir(parents=True)
    (loc_dir / "card.toml").write_text('description = "A small room."\n')

    # Scenes.
    plot_dir = tmp / "assets" / "plot" / "e2e"
    plot_dir.mkdir(parents=True)
    (plot_dir / "ini_scene.toml").write_text(
        '''id = "ini_scene"
name = "Start"
language = "English"
zeitgeist = "test"
tone = "neutral"
world = "test_world"

[character]
pool = ["Player", "Narrator", "NPC"]
inits = ["Player", "Narrator", "NPC"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["room"]
init = "room"

[plot]
considerations = ""
scene = "The beginning."

[plot.next]
considerations = "None"

[plot.next.scene2]
desc = "Move on"
'''
    )
    (plot_dir / "scene2.toml").write_text(
        '''id = "scene2"
name = "End"
language = "English"
zeitgeist = "test"
tone = "neutral"
type = "fin"

[character]
pool = ["Player", "Narrator", "NPC"]
inits = ["Player", "Narrator", "NPC"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["room"]
init = "room"

[plot]
considerations = ""
scene = "The end."
'''
    )
    return plot_dir / "ini_scene.toml"


def _run_aractl(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "ara.cli.aractl"] + args
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def aractl_server():
    """Start an AgentServer on a temp socket and yield (tmp, socket_path, scene_path)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        scene_path = _write_assets(tmp)

        # Ara requires these subdirectories.
        (tmp / "chroma").mkdir(exist_ok=True)
        (tmp / "saves").mkdir(exist_ok=True)
        (tmp / "sockets").mkdir(exist_ok=True)

        settings = AraSettings(
            data_dir=tmp,
            api_key="",
            api_endpoint="",
            api_model="",
        )
        db = MagicMock(spec=ChromaStore)

        # Pre-canned LLM responses:
        # 1) narrator turn generation
        # 2) NPC end-of-scene scratch update
        # 3) scene-transition summarizer
        # 4) finalize-turn narrator text
        responses = [
            StreamResult(content="Narrator speaks."),
            StreamResult(content=""),
            StreamResult(
                content="SUMMARY Player:\nStill in the room.\n\n"
                        "SUMMARY Narrator:\nObserving quietly.\n\n"
                        "SUMMARY NPC:\nWaiting.\n\n"
                        "LOCATION:\nA small room.\n\n"
                        "TIME:\nafternoon"
            ),
            StreamResult(content="The scene fades."),
        ]
        mock_client = MockLLMClient(responses)
        story = Story(settings, db, mock_client, scene_path)

        # Mock orchestrator decisions once the scene is loaded.
        decide_state = {"call": 0}

        def _decide(**_kwargs: object) -> TurnDecision:
            scene = story.current_scene
            assert scene is not None, "scene not loaded before orchestrator decision"
            call = decide_state["call"]
            decide_state["call"] += 1
            if call == 0:
                return TurnDecision(
                    next_char=scene.narrator,
                    directive="Set the scene",
                    suggestions=[],
                    entering_chars=set(),
                    exiting_chars=set(),
                    switch_location=None,
                    next_scene=None,
                )
            if call == 1:
                return TurnDecision(
                    next_char=scene.player,
                    directive="",
                    suggestions=["Go to scene2"],
                    entering_chars=set(),
                    exiting_chars=set(),
                    switch_location=None,
                    next_scene=None,
                )
            return TurnDecision(
                next_char=scene.player,
                directive="",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                next_scene="scene2",
            )

        original_orchestrator = story.engine.orchestrator
        original_orchestrator.decide_next_turn = _decide  # type: ignore[method-assign]

        socket_path = str(tmp / "sockets" / "aractl_e2e.sock")
        server = AgentServer(story, socket_path=socket_path, client_step=1)
        thread = threading.Thread(target=server.start_listening, daemon=True)
        thread.start()
        for _ in range(40):
            if os.path.exists(socket_path):
                break
            time.sleep(0.05)

        yield tmp, socket_path, scene_path, story

        server.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


class TestAractlE2E:
    def test_start_and_step(self, aractl_server) -> None:
        """aractl start + step loads the scene and produces a turn event."""
        tmp, socket_path, scene_path, story = aractl_server

        _run_aractl(["--socket", socket_path, "start"], tmp)

        result = _run_aractl(["--json", "--socket", socket_path, "step"], tmp)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["event"] == "scene_loaded"
        assert data["scene"]["id"] == "ini_scene"

        result = _run_aractl(["--json", "--socket", socket_path, "step"], tmp)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["event"] == "turn"
        assert data["speaker"] == "Narrator"

    def test_full_walkthrough(self, aractl_server) -> None:
        """Drive a complete scene transition through the CLI."""
        tmp, socket_path, scene_path, story = aractl_server

        events: list[dict] = []

        def _step() -> dict:
            result = _run_aractl(["--json", "--socket", socket_path, "step"], tmp)
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)

        _run_aractl(["--socket", socket_path, "start"], tmp)

        events.append(_step())  # scene_loaded
        events.append(_step())  # narrator turn
        events.append(_step())  # needs_player_input

        result = _run_aractl(["--json", "--socket", socket_path, "reply", "Let's go"], tmp)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data.get("submitted") == "Let's go"

        events.append(_step())  # transition
        events.append(_step())  # story_complete for scene2 (fin scene loads then ends)

        assert events[0]["event"] == "scene_loaded"
        assert events[1]["event"] == "turn"
        assert events[2]["event"] == "needs_player_input"
        assert events[2]["suggestions"] == ["Go to scene2"]
        assert events[3]["event"] == "transition"
        assert events[3]["next_scene"] == "scene2"
        assert events[4]["event"] == "story_complete"

        # State endpoint should reflect the completed story.
        result = _run_aractl(["--json", "--socket", socket_path, "state"], tmp)
        assert result.returncode == 0, result.stderr
        state = json.loads(result.stdout)
        assert state["story"]["finished"] is True
        assert "ini_scene" in state["story"]["scene_history"]
        assert "scene2" in state["story"]["scene_history"]
