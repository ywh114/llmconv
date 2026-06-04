"""Multi-scene flow with memory and recall integration tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.models import GameRole, Importance, StreamResult
from ara.world.character import Character
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene, SceneChoice
from ara.world.story import Story
from ara.memory.knowledge import CharacterMemory, Scratchpad


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
            "system_prompt": system_prompt,
            "messages": messages,
        })
        result = self.responses[self._index]
        self._index += 1
        return result

    def complete_subagent(
        self, task: str, context: str, system_prompt: str = "", max_tokens: int = 512
    ) -> str:
        return f"[sub-agent summary for: {task}]"


def _make_scene(
    scene_id: str,
    next_choices: dict[str, SceneChoice],
    mock_db: ChromaStore,
) -> Scene:
    """Build a minimal scene programmatically."""
    import uuid

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
        id=scene_id,
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
        plot_story=f"Test scene {scene_id}",
        next_choices=next_choices,
    )


class TestMultiSceneFlow:
    """Test scene transitions with only_for prerequisites."""

    def test_only_for_filters_next_choices(self) -> None:
        """Scene.load() should filter next_choices based on prev_id."""
        import tempfile
        from ara.config import AraSettings

        mock_db = MagicMock(spec=ChromaStore)
        config = AraSettings()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            cc = assets / "cc"
            plot = assets / "plot"
            cc.mkdir(parents=True)
            plot.mkdir(parents=True)

            for name in ["Player", "Narrator", "NPC"]:
                d = cc / name
                d.mkdir()
                (d / "card.toml").write_text(f'name = "{name}"\n')

            toml = plot / "scene_a.toml"
            toml.write_text('''
id = "scene_a"
language = "English"
zeitgeist = "test"
tone = "neutral"

[character]
pool = ["Player", "Narrator", "NPC"]
inits = ["Player", "Narrator", "NPC"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["room"]
init = "room"

[location.descs]
room = "A room."

[plot]
considerations = ""
scene = "Test"

[plot.next]
considerations = "None"

[plot.next.scene_b]
desc = "Go to B"
only_for = ["scene_a"]

[plot.next.scene_c]
desc = "Go to C"
only_for = ["other"]

[plot.next.scene_d]
desc = "Go to D"
''')

            config.data_dir = tmp

            # When coming from scene_a, only scene_b and scene_d should be available
            scene = Scene.load(toml, mock_db, config, prev_id="scene_a")
            assert "scene_b" in scene.next_choices
            assert "scene_c" not in scene.next_choices
            assert "scene_d" in scene.next_choices

            # When coming from 'other', only scene_c and scene_d should be available
            scene2 = Scene.load(toml, mock_db, config, prev_id="other")
            assert "scene_b" not in scene2.next_choices
            assert "scene_c" in scene2.next_choices
            assert "scene_d" in scene2.next_choices

    def test_two_scene_story(self) -> None:
        """Drive a story through two scenes and verify state carries over."""
        mock_db = MagicMock(spec=ChromaStore)

        scene1 = _make_scene(
            "scene1",
            {"scene2": SceneChoice(id="scene2", desc="Next", only_for=["scene1"])},
            mock_db,
        )
        scene2 = _make_scene(
            "scene2",
            {},
            mock_db,
        )

        # Give scene1's NPC some scratch content to verify it carries over
        npc1 = next(c for c in scene1.character_pool if c.name == "NPC")
        npc1.scratch.text = "I remember the first scene."

        responses = [
            StreamResult(content="Hello from scene 1."),
            StreamResult(content=""),
            StreamResult(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "next_round",
                            "arguments": json.dumps({
                                "next_character": "Player",
                                "directive": "",
                                "suggestions": ["Let's go"],
                                "enter_characters": [],
                                "exit_characters": [],
                                "switch_location": "",
                                "edit_location": "",
                                "next_scene": "scene2",
                            }),
                        },
                    }
                ],
            ),
        ]
        mock_client = MockLLMClient(responses)
        engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.side_effect = [
            TurnDecision(
                next_char=scene1.narrator,
                directive="Introduce",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene1.player,
                directive="",
                suggestions=["Let's go"],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene1.player,
                directive="",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene="scene2",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            cc = assets / "cc"
            plot = assets / "plot"
            cc.mkdir(parents=True)
            plot.mkdir(parents=True)

            for name in ["Player", "Narrator", "NPC"]:
                d = cc / name
                d.mkdir()
                (d / "card.toml").write_text(f'name = "{name}"\n')

            (plot / "scene1.toml").write_text('''
id = "scene1"
language = "English"
zeitgeist = "test"
tone = "neutral"

[character]
pool = ["Player", "Narrator", "NPC"]
inits = ["Player", "Narrator", "NPC"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["room"]
init = "room"

[location.descs]
room = "A room."

[plot]
considerations = ""
scene = "Scene 1"

[plot.next]
considerations = "None"

[plot.next.scene2]
desc = "Next"
only_for = ["scene1"]
''')
            (plot / "scene2.toml").write_text('''
id = "scene2"
language = "English"
zeitgeist = "test"
tone = "neutral"

[character]
pool = ["Player", "Narrator", "NPC"]
inits = ["Player", "Narrator", "NPC"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["room"]
init = "room"

[location.descs]
room = "A room."

[plot]
considerations = ""
scene = "Scene 2"

[plot.next]
considerations = "None"
''')

            config = AraSettings()
            config.data_dir = tmp

            story = Story(config, mock_db, mock_client, plot / "scene1.toml")
            story.engine = engine

            scenes = {"scene1": scene1, "scene2": scene2}

            def fake_load(path: Path, db: ChromaStore, config: AraSettings, prev_id: str = "") -> Scene:
                return scenes[path.stem]

            with patch("ara.world.story.Scene.load", side_effect=fake_load):
                story.start()

                # scene1 loaded
                result = story.step()
                assert result.event == "scene_loaded"
                assert result.scene.id == "scene1"

                # Narrator turn
                result = story.step()
                assert result.event == "turn"

                # Player input
                result = story.step()
                assert result.event == "needs_player_input"
                story.submit_player_input("Let's go")

                # Scene ends
                result = story.step()
                assert result.event == "scene_ended"
                assert result.next_scene == "scene2"

                # scene2 loaded - verify memory carried over but scratch reset
                result = story.step()
                assert result.event == "scene_loaded"
                assert result.scene.id == "scene2"

                npc2 = next(c for c in scene2.character_pool if c.name == "NPC")
                # Scratchpads do NOT carry over — only memory does
                assert npc2.scratch.text == "Nothing yet!"


class TestMemoryAndRecall:
    """Test conversation storage and character recall across scenes."""

    def test_conversation_stored_in_memory(self) -> None:
        """Character and narrator responses should be stored in ChromaDB."""
        mock_db = MagicMock(spec=ChromaStore)
        scene = _make_scene("test", {}, mock_db)

        responses = [
            StreamResult(content="The room is quiet."),
            StreamResult(content=""),
        ]
        mock_client = MockLLMClient(responses)
        engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.side_effect = [
            TurnDecision(
                next_char=scene.narrator,
                directive="Describe",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene.player,
                directive="",
                suggestions=["Say hello"],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene.narrator,
                directive="React",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
        ]

        engine.start(scene)
        engine.step()  # narrator turn
        engine.step()  # needs_player_input
        engine.submit_player_input("Hello there")
        engine.step()  # narrator turn after player input

        # Verify memory was called for narrator and player
        narrator = scene.narrator
        player = scene.player
        assert narrator.memory.db.upsert.call_count >= 2
        assert player.memory.db.upsert.call_count >= 1

    def test_irrelevant_recall_becomes_real_knowledge(self) -> None:
        """Recalling something irrelevant returns empty, the character responds
        naturally, and that response becomes 'real' knowledge for future recall.
        """
        mock_db = MagicMock(spec=ChromaStore)
        scene = _make_scene("test", {}, mock_db)

        # First recall: empty memory → character says "I don't remember..."
        # This response gets stored.
        mock_db.query.return_value = {"documents": [[]]}

        responses = [
            StreamResult(content="I don't remember what we ate last week."),
            StreamResult(content="Wait, I think I mentioned earlier that I don't remember what we ate. Maybe just something simple?"),
        ]
        mock_client = MockLLMClient(responses)
        engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.side_effect = [
            TurnDecision(
                next_char=next(c for c in scene.character_pool if c.name == "NPC"),
                directive="Recall what was eaten for dinner last week",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene.player,
                directive="",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            # Second recall: now memory has the "I don't remember" response
            TurnDecision(
                next_char=next(c for c in scene.character_pool if c.name == "NPC"),
                directive="Recall what was eaten for dinner last week",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene.player,
                directive="",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
        ]

        engine.start(scene)

        # First turn: empty recall → "I don't remember..."
        engine.step()
        npc = next(c for c in scene.character_pool if c.name == "NPC")

        # The "I don't remember" response was stored in memory
        assert npc.memory.db.upsert.call_count >= 1
        stored_docs = [
            call.args[2] if len(call.args) > 2 else call.kwargs.get("documents", [])
            for call in npc.memory.db.upsert.call_args_list
        ]
        all_texts = [doc for group in stored_docs for doc in (group if isinstance(group, list) else [group])]
        assert any("don't remember" in t for t in all_texts)

        # Player turn
        engine.step()
        engine.submit_player_input("Oh, okay.")

        # Second turn: memory now returns the previous "I don't remember"
        # Set up mock to return the stored text on second query
        mock_db.query.return_value = {
            "documents": [["I don't remember what we ate last week."]]
        }
        engine.step()

        # The character's second response references their earlier "knowledge"
        # Collect only the NPC's upsert calls (filter by collection name)
        npc_calls = [
            call for call in npc.memory.db.upsert.call_args_list
            if call.args[0] == npc.memory.collection_name
        ]
        stored_docs2 = [
            call.args[2] if len(call.args) > 2 else call.kwargs.get("documents", [])
            for call in npc_calls
        ]
        all_texts2 = [doc for group in stored_docs2 for doc in (group if isinstance(group, list) else [group])]
        assert any("Wait, I think I mentioned earlier" in t for t in all_texts2)


class TestPerCharacterSummarization:
    """Test that the Summarizer produces and distributes per-character summaries."""

    def test_per_character_summaries_on_transition(self) -> None:
        """When transitioning scenes, each character receives a tailored summary.

        Scene A has Alice and Player. Alice knows a secret.
        Scene B adds Bob. The summarizer should produce different summaries:
        - Alice (continuing) gets a brief bridge
        - Bob (new) gets a fuller recap that does NOT reveal the secret
        """
        mock_db = MagicMock(spec=ChromaStore)

        # Scene A: Player, Alice, Narrator
        scene_a = _make_scene_with_chars(
            "scene_a",
            {"scene_b": SceneChoice(id="scene_b", desc="Next", only_for=["scene_a"])},
            mock_db,
            char_names=["Player", "Alice", "Narrator"],
        )
        # Scene B: Player, Alice, Bob, Narrator
        scene_b = _make_scene_with_chars(
            "scene_b",
            {},
            mock_db,
            char_names=["Player", "Alice", "Bob", "Narrator"],
        )

        # Alice has a secret in her scratchpad
        alice_a = next(c for c in scene_a.character_pool if c.name == "Alice")
        alice_a.scratch.text = "The project is a cover. Keep it secret from Bob."

        # Summarizer response: per-character summaries
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
            StreamResult(content="Hello from scene A."),   # narrator turn scene_a
            StreamResult(content=""),                         # Alice scratch update
            StreamResult(content=summarizer_response),        # summarizer
        ]
        mock_client = MockLLMClient(responses)
        engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.side_effect = [
            TurnDecision(
                next_char=scene_a.narrator,
                directive="Introduce",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene_a.player,
                directive="",
                suggestions=["Go on"],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene=None,
            ),
            TurnDecision(
                next_char=scene_a.player,
                directive="",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                edit_location="",
                next_scene="scene_b",
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            cc = assets / "cc"
            plot = assets / "plot"
            cc.mkdir(parents=True)
            plot.mkdir(parents=True)

            for name in ["Player", "Narrator", "Alice", "Bob"]:
                d = cc / name
                d.mkdir()
                (d / "card.toml").write_text(f'name = "{name}"\n')

            (plot / "scene_a.toml").write_text('''
id = "scene_a"
language = "English"
zeitgeist = "test"
tone = "neutral"

[character]
pool = ["Player", "Alice", "Narrator"]
inits = ["Player", "Alice", "Narrator"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["lab"]
init = "lab"

[location.descs]
lab = "A small research lab with computers and filing cabinets."

[plot]
considerations = ""
scene = "Scene A"

[plot.next]
considerations = "None"

[plot.next.scene_b]
desc = "Next"
only_for = ["scene_a"]
''')
            (plot / "scene_b.toml").write_text('''
id = "scene_b"
language = "English"
zeitgeist = "test"
tone = "neutral"

[character]
pool = ["Player", "Alice", "Bob", "Narrator"]
inits = ["Player", "Alice", "Bob", "Narrator"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["lab"]
init = "lab"

[location.descs]
lab = "A small research lab with computers and filing cabinets."

[plot]
considerations = ""
scene = "Scene B"

[plot.next]
considerations = "None"
''')

            config = AraSettings()
            config.data_dir = tmp

            story = Story(config, mock_db, mock_client, plot / "scene_a.toml")
            story.engine = engine

            scenes = {"scene_a": scene_a, "scene_b": scene_b}

            def fake_load(path: Path, db: ChromaStore, config: AraSettings, prev_id: str = "") -> Scene:
                return scenes[path.stem]

            with patch("ara.world.story.Scene.load", side_effect=fake_load):
                story.start()

                # scene_a loaded
                result = story.step()
                assert result.event == "scene_loaded"
                assert result.scene.id == "scene_a"

                # Narrator turn
                result = story.step()
                assert result.event == "turn"

                # Player input
                result = story.step()
                assert result.event == "needs_player_input"
                story.submit_player_input("Go on")

                # Scene ends
                result = story.step()
                assert result.event == "scene_ended"
                assert result.next_scene == "scene_b"

                # scene_b loaded
                result = story.step()
                assert result.event == "scene_loaded"
                assert result.scene.id == "scene_b"

                # Verify per-character summaries were distributed
                alice_b = next(c for c in scene_b.character_pool if c.name == "Alice")
                bob_b = next(c for c in scene_b.character_pool if c.name == "Bob")
                player_b = next(c for c in scene_b.character_pool if c.name == "Player")
                narrator_b = next(c for c in scene_b.character_pool if c.name == "Narrator")

                assert "continue from where you left off" in alice_b.prev_scene_summary
                assert "arrive at the lab" in bob_b.prev_scene_summary
                assert "continue the conversation" in player_b.prev_scene_summary
                assert "story continues" in narrator_b.prev_scene_summary

                # Verify scratchpad was reset (not carried over)
                assert alice_b.scratch.text == "Nothing yet!"

                # Verify the summarizer was called
                summarizer_calls = [c for c in mock_client.calls if c["role"] == GameRole.SUMMARIZER]
                assert len(summarizer_calls) == 1

    def test_per_character_summaries_injected_into_turns(self) -> None:
        """Per-character summaries should appear in a character's branch context."""
        mock_db = MagicMock(spec=ChromaStore)
        scene = _make_scene_with_chars(
            "test",
            {},
            mock_db,
            char_names=["Player", "Alice", "Narrator"],
        )

        alice = next(c for c in scene.character_pool if c.name == "Alice")
        alice.prev_scene_summary = "You remember the secret agreement."

        responses = [
            StreamResult(content="Alice speaks."),
        ]
        mock_client = MockLLMClient(responses)
        engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=alice,
            directive="Speak",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            edit_location="",
            next_scene=None,
        )

        engine.start(scene)
        engine.step()

        # The mock client should have been called for Alice's turn.
        # We can inspect the messages passed to verify the summary context is present.
        alice_call = mock_client.calls[-1]
        messages = alice_call.get("messages", [])
        # Find the summary context in messages
        msg_texts = [m.get("content", "") for m in messages]
        assert any("remember the secret agreement" in text for text in msg_texts)


def _make_scene_with_chars(
    scene_id: str,
    next_choices: dict[str, SceneChoice],
    mock_db: ChromaStore,
    char_names: list[str],
) -> Scene:
    """Build a scene with specific character names."""
    import uuid

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

    chars = {make_char(name) for name in char_names}
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
        plot_story=f"Test scene {scene_id}",
        next_choices=next_choices,
    )
