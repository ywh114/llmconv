"""Mock server for manually testing per-character summarization via aractl."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ara.agent.server import AgentServer
from ara.config import AraSettings
from ara.memory.chroma import ChromaStore
from ara.models import GameRole, Importance, StreamResult
from ara.world.character import Character
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene, SceneChoice
from ara.world.story import Story
from ara.memory.knowledge import CharacterMemory, Scratchpad

import uuid


def make_char(name: str, mock_db) -> Character:
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


def make_scene(scene_id, next_choices, mock_db, char_names):
    chars = {make_char(name, mock_db) for name in char_names}
    player = next(c for c in chars if c.name == "Player")
    narrator = next(c for c in chars if c.name == "Narrator")
    loc = Location(name="lab", desc="A lab.")
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
        next_choices=next_choices,
    )


class MockLLMClient:
    def __init__(self, responses):
        self.responses = responses
        self._index = 0
        self.calls = []

    def complete(self, role, system_prompt, messages, tools=None, tool_choice=None, stream=True, print_stream=False):
        self.calls.append({"role": role, "tools": tools, "tool_choice": tool_choice, "messages": messages, "system_prompt": system_prompt})
        result = self.responses[self._index]
        self._index += 1
        return result

    def complete_subagent(self, task, context, system_prompt="", max_tokens=512):
        return f"[sub-agent summary for: {task}]"


def main():
    mock_db = MagicMock(spec=ChromaStore)

    scene_a = make_scene("scene_a", {"scene_b": SceneChoice(id="scene_b", desc="Next", only_for=["scene_a"])}, mock_db, ["Player", "Alice", "Narrator"])
    scene_b = make_scene("scene_b", {}, mock_db, ["Player", "Alice", "Bob", "Narrator"])

    alice_a = next(c for c in scene_a.character_pool if c.name == "Alice")
    alice_a.scratch.text = "The project is a cover. Keep it secret from Bob."

    summarizer_response = (
        "SUMMARY Alice:\n"
        "You continue from where you left off, still guarding the secret.\n\n"
        "SUMMARY Bob:\n"
        "You arrive at the lab where Alice and Player are working. They seem focused.\n\n"
        "SUMMARY Player:\n"
        "You continue the conversation with Alice.\n\n"
        "SUMMARY Narrator:\n"
        "The story continues in the lab.\n\n"
        "LOCATION:\n"
        "A small research lab with computers and filing cabinets."
    )

    responses = [
        StreamResult(content="Hello from scene A."),
        StreamResult(content=""),
        StreamResult(content=summarizer_response),
        StreamResult(content="Hello from scene B."),
    ]
    mock_client = MockLLMClient(responses)
    engine = Engine(mock_client, db=mock_db)
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.side_effect = [
        TurnDecision(next_char=scene_a.narrator, directive="Introduce", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, edit_location="", next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=["Go on"], entering_chars=set(), exiting_chars=set(), switch_location=None, edit_location="", next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, edit_location="", next_scene="scene_b"),
        TurnDecision(next_char=scene_b.narrator, directive="Introduce", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, edit_location="", next_scene=None),
    ]

    config = AraSettings()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        plot = tmp / "plot"
        plot.mkdir()

        # Write dummy TOML files so _load_scene passes the exists() check
        (plot / "scene_a.toml").write_text('id = "scene_a"\n')
        (plot / "scene_b.toml").write_text('id = "scene_b"\n')

        story = Story(config, mock_db, mock_client, plot / "scene_a.toml")
        story.engine = engine

        scenes = {"scene_a": scene_a, "scene_b": scene_b}

        def fake_load(path, db, config, prev_id=""):
            return scenes[path.stem]

        with patch("ara.world.story.Scene.load", side_effect=fake_load):
            story.start()
            story.step()  # scene_a loaded
            story.step()  # narrator turn
            story.step()  # needs_player_input
            story.submit_player_input("Go on")
            story.step()  # scene_ended
            story.step()  # scene_b loaded
            story.step()  # narrator turn in scene_b

        print("=== Per-character summaries after scene transition ===")
        for char in story.current_scene.character_pool:
            print(f"\n{char.name}:")
            print(f"  prev_scene_summary: {char.prev_scene_summary!r}")
            print(f"  scratch: {char.scratch.text!r}")

        # Start the agent server so aractl can connect
        socket_path = "sockets/ara_agent_test.sock"
        os.makedirs("sockets", exist_ok=True)
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        server = AgentServer(story, socket_path=socket_path)
        print(f"\n[Mock server] Listening on {socket_path}")
        print("Use: python examples/aractl.py --socket sockets/ara_agent_test.sock state")
        print("     python examples/aractl.py --socket sockets/ara_agent_test.sock debug summary Alice")
        print("     python examples/aractl.py --socket sockets/ara_agent_test.sock debug summary Bob")
        print("Press Ctrl+C to stop.")
        try:
            server.start_listening()
        except KeyboardInterrupt:
            print("\n[Mock server] Shutting down.")
            server.shutdown()


if __name__ == "__main__":
    main()
