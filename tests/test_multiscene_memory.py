"""Multi-scene flow with memory and recall integration tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


from ara.config import AraSettings
from ara.memory.chroma import ChromaStore
from ara.llm.models import GameRole, StreamResult
from ara.world.character import Character
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene, SceneChoice, load_location
from ara.world.story import Story
from ara.world.orchestrator import Orchestrator
from ara.world.summarizer import Summarizer, SceneStateModifiers
from ara.world.character import load_character
from ara.persistence.save import SaveManager

from tests.helpers import ScriptedLLMClient as MockLLMClient
from tests.helpers import make_scene


def _make_scene(
    scene_id: str,
    next_choices: dict[str, SceneChoice],
    mock_db: ChromaStore,
) -> Scene:
    """Build a minimal scene programmatically."""
    return make_scene(scene_id, mock_db, next_choices=next_choices)


class TestMultiSceneFlow:
    """Test scene transitions with only_for prerequisites."""

    def test_prereq_scenes_filters_next_choices(self) -> None:
        """Scene.load() should filter next_choices based on prev_id."""
        import tempfile
        from ara.config import AraSettings

        mock_db = MagicMock(spec=ChromaStore)
        config = AraSettings()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            story = "prereq_test"
            cc = assets / "cc" / story
            plot = assets / "plot" / story
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
prereq_scenes = ["scene_a"]

[plot.next.scene_c]
desc = "Go to C"
prereq_scenes = ["other"]

[plot.next.scene_d]
desc = "Go to D"
''')

            config.data_dir = tmp

            # When coming from scene_a, only scene_b and scene_d should be available
            scene = Scene.load(toml, mock_db, config, scene_history=["scene_a"])
            assert "scene_b" in scene.next_choices
            assert "scene_c" not in scene.next_choices
            assert "scene_d" in scene.next_choices

            # With "other" in history, all choices are available because
            # prereq_scenes checks visited history (which includes the current scene).
            scene2 = Scene.load(toml, mock_db, config, scene_history=["other"])
            assert "scene_b" in scene2.next_choices
            assert "scene_c" in scene2.next_choices
            assert "scene_d" in scene2.next_choices

    def test_two_scene_story(self) -> None:
        """Drive a story through two scenes and verify state carries over."""
        mock_db = MagicMock(spec=ChromaStore)

        scene1 = _make_scene(
            "scene1",
            {"scene2": SceneChoice(id="scene2", desc="Next", prereq_scenes=["scene1"])},
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
            story = "two_scene_test"
            cc = assets / "cc" / story
            plot = assets / "plot" / story
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
prereq_scenes = ["scene1"]
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

            def fake_load(path: Path, db: ChromaStore, config: AraSettings, scene_history: list[str] | None = None, **kwargs) -> Scene:
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
                assert result.event == "transition"
                assert result.phase == "ended"
                assert result.next_scene == "scene2"

                # scene2 finalised and loaded - verify memory carried over but scratch reset
                result = story.step()
                assert result.event == "scene_loaded"
                assert result.scene.id == "scene2"

                npc2 = next(c for c in scene2.character_pool if c.name == "NPC")
                # Scratchpads do NOT carry over - only memory does
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
            {"scene_b": SceneChoice(id="scene_b", desc="Next", prereq_scenes=["scene_a"])},
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
            story = "summary_test"
            cc = assets / "cc" / story
            plot = assets / "plot" / story
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
prereq_scenes = ["scene_a"]
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

            def fake_load(path: Path, db: ChromaStore, config: AraSettings, scene_history: list[str] | None = None, **kwargs) -> Scene:
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
                assert result.event == "transition"
                assert result.phase == "ended"
                assert result.next_scene == "scene_b"

                # scene_b finalised and loaded
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
    return make_scene(
        scene_id,
        mock_db,
        char_names=tuple(char_names),
        next_choices=next_choices,
        location_name="lab",
    )
class _FakeCollection:
    """Minimal in-memory ChromaDB collection for save/load tests."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def upsert(self, *, ids: list[str], documents: list[str], metadatas: list[Any] | None = None) -> None:
        metadatas = metadatas or [{} for _ in ids]
        for i, doc_id in enumerate(ids):
            self.docs[doc_id] = {
                "document": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
            }

    def get(self, where: Any | None = None) -> dict[str, Any]:
        return {
            "ids": list(self.docs.keys()),
            "documents": [d["document"] for d in self.docs.values()],
            "metadatas": [d["metadata"] for d in self.docs.values()],
        }

    def delete(self, *, ids: list[str]) -> None:
        for doc_id in ids:
            self.docs.pop(doc_id, None)

    def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
        where: Any | None = None,
    ) -> dict[str, Any]:
        docs = [d["document"] for d in self.docs.values()]
        per_query = docs[:n_results]
        return {
            "documents": [per_query for _ in query_texts],
            "ids": [list(self.docs.keys())[:n_results] for _ in query_texts],
            "metadatas": [[d["metadata"] for d in list(self.docs.values())[:n_results]] for _ in query_texts],
            "distances": [[0.1 for _ in per_query] for _ in query_texts],
        }


class _FakeChromaStore:
    """Minimal ChromaDB stand-in that avoids heavy dependencies in tests."""

    def __init__(self) -> None:
        self._collections: dict[str, _FakeCollection] = {}

    def collection(self, name: str) -> _FakeCollection:
        if name not in self._collections:
            self._collections[name] = _FakeCollection()
        return self._collections[name]

    def get_or_create_collection(self, **kwargs: Any) -> _FakeCollection:
        name = kwargs.get("name", "default")
        return self.collection(name)

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        metadatas: Any | None = None,
    ) -> None:
        self.collection(collection_name).upsert(
            ids=ids, documents=documents, metadatas=metadatas
        )

    def query(self, collection_name: str, query_texts: list[str], n_results: int = 5, where: Any | None = None) -> dict[str, Any]:
        return self.collection(collection_name).query(
            query_texts=query_texts, n_results=n_results, where=where
        )

    def get_all(self, collection_name: str, where: Any | None = None) -> dict[str, Any]:
        return self.collection(collection_name).get(where=where)

    def clear_all_collections(self) -> None:
        self._collections.clear()


class TestCrossSceneLiveCacheAndMemory:
    """Tests for live-cache authority, orchestrator digest, and save/load."""

    def test_live_cache_identity_across_scenes(self) -> None:
        """Scene.load() should reuse live-cache objects for shared characters/locations."""
        mock_db = MagicMock(spec=ChromaStore)
        config = AraSettings()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            story_name = "live_cache_test"
            cc = assets / "cc" / story_name
            lc = assets / "lc" / story_name
            plot = assets / "plot" / story_name
            cc.mkdir(parents=True)
            lc.mkdir(parents=True)
            plot.mkdir(parents=True)

            for name in ["Player", "Narrator", "NPC"]:
                d = cc / name
                d.mkdir()
                (d / "card.toml").write_text(f'name = "{name}"\n')

            (lc / "room").mkdir()
            (lc / "room" / "card.toml").write_text(
                'name = "room"\ndescription = "A room."\n'
            )

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

[plot]
considerations = ""
scene = "Scene 1"
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

[plot]
considerations = ""
scene = "Scene 2"
''')
            config.data_dir = tmp

            live_chars: dict[str, Character] = {}
            live_locs: dict[str, Location] = {}

            scene1 = Scene.load(
                plot / "scene1.toml",
                mock_db,
                config,
                live_characters=live_chars,
                live_locations=live_locs,
            )
            npc = scene1.character_by_name("NPC")
            assert npc is not None
            npc.status = {"mood": "worried"}
            room = scene1.location_by_name("room")
            assert room is not None
            room.desc = "A dusty room."

            scene2 = Scene.load(
                plot / "scene2.toml",
                mock_db,
                config,
                live_characters=live_chars,
                live_locations=live_locs,
            )
            npc2 = scene2.character_by_name("NPC")
            room2 = scene2.location_by_name("room")

            assert npc2 is npc
            assert npc2.status == {"mood": "worried"}
            assert room2 is room
            assert room2.desc == "A dusty room."

    def test_orchestrator_character_scratch_digest(self) -> None:
        """_character_scratch_digest should expose non-default scratches only."""
        mock_db = MagicMock(spec=ChromaStore)
        scene = _make_scene("digest", {}, mock_db)

        npc = next(c for c in scene.character_pool if c.name == "NPC")
        npc.scratch.text = "secret plan to flee"
        player = scene.player
        player.scratch.text = "Nothing yet!"
        narrator = scene.narrator
        narrator.scratch.text = ""

        digest = Orchestrator._character_scratch_digest(scene)

        assert len(digest) == 2
        assert digest[0]["role"] == "user"
        assert digest[1]["role"] == "assistant"
        assert "secret plan to flee" in digest[1]["content"]
        assert "Nothing yet!" not in digest[1]["content"]
        assert digest[1].get("_canonical_name") == "__orchestrator__"

    def test_summarizer_receives_history_context(self) -> None:
        """The summarizer prompt should include recalled story history."""
        mock_db = MagicMock(spec=ChromaStore)
        scene = _make_scene("history", {}, mock_db)

        response = """SUMMARY NPC:
The NPC arrives at the inn.

LOCATION:
A room.

TIME: afternoon"""
        client = MockLLMClient([StreamResult(content=response)])
        summarizer = Summarizer(client)

        summarizer.summarize_transition(
            current_scene=scene,
            current_scene_considerations="",
            next_scene_plot="The hero arrives at the inn.",
            next_scene_considerations="",
            conversation_context=[],
            location_desc="A room.",
            language="English",
            scratchpads={},
            next_scene_chars=["NPC"],
            history_context="Earlier scene: a promise was made at the crossroads.",
        )

        assert len(client.calls) == 1
        prompt = client.calls[0]["messages"][0]["content"]
        assert "Earlier scene: a promise was made at the crossroads." in prompt
        assert "Relevant summaries from earlier scenes:" in prompt

    def test_save_load_preserves_live_cache(self) -> None:
        """Save/load should keep off-screen live-cache characters and locations."""
        fake_db = _FakeChromaStore()
        mock_client = MockLLMClient([])
        config = AraSettings()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            story_name = "save_cache_test"
            cc = assets / "cc" / story_name
            lc = assets / "lc" / story_name
            plot = assets / "plot" / story_name
            cc.mkdir(parents=True)
            lc.mkdir(parents=True)
            plot.mkdir(parents=True)

            for name in ["Player", "Narrator", "NPC"]:
                d = cc / name
                d.mkdir()
                (d / "card.toml").write_text(f'name = "{name}"\n')

            off_char_dir = cc / "OffChar"
            off_char_dir.mkdir()
            (off_char_dir / "card.toml").write_text('name = "OffChar"\n')

            (lc / "room").mkdir()
            (lc / "room" / "card.toml").write_text('name = "room"\n')
            off_loc_dir = lc / "OffLoc"
            off_loc_dir.mkdir()
            (off_loc_dir / "card.toml").write_text('name = "OffLoc"\n')

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

[plot]
considerations = ""
scene = "Scene 1"
''')
            (plot / "scene2.toml").write_text('''
id = "scene2"
language = "English"
zeitgeist = "test"
tone = "neutral"

[character]
pool = ["Player", "Narrator", "NPC", "OffChar"]
inits = ["Player", "Narrator", "NPC", "OffChar"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["room", "OffLoc"]
init = "room"

[plot]
considerations = ""
scene = "Scene 2"
''')
            config.data_dir = tmp

            story = Story(config, fake_db, mock_client, plot / "scene1.toml")
            scene = Scene.load(
                plot / "scene1.toml",
                fake_db,
                config,
                registry=story.registry,
                live_characters=story._live_characters,
                live_locations=story._live_locations,
            )
            story._current_scene = scene

            off_char = load_character(off_char_dir, fake_db, "en")
            off_char.scratch.text = "offscreen scratch"
            off_char.status = {"hp": 5}
            story._live_characters["OffChar"] = off_char

            off_loc = load_location(off_loc_dir, language="en")
            off_loc.desc = "offscreen description"
            story._live_locations["OffLoc"] = off_loc

            manager = SaveManager(config)
            snapshot = manager._build_snapshot(story)

            fresh_story = Story(config, fake_db, mock_client, plot / "scene1.toml")

            with patch("ara.world.story.Engine.start"), patch(
                "ara.world.story.Summarizer.apply_initial_state_modifiers",
                return_value=SceneStateModifiers(),
            ):
                manager._apply_snapshot(fresh_story, snapshot)

            assert "OffChar" in fresh_story._live_characters
            restored_char = fresh_story._live_characters["OffChar"]
            assert restored_char.scratch.text == "offscreen scratch"
            assert restored_char.status == {"hp": 5}

            assert "OffLoc" in fresh_story._live_locations
            restored_loc = fresh_story._live_locations["OffLoc"]
            assert restored_loc.desc == "offscreen description"

            scene2 = Scene.load(
                plot / "scene2.toml",
                fake_db,
                config,
                registry=fresh_story.registry,
                live_characters=fresh_story._live_characters,
                live_locations=fresh_story._live_locations,
            )
            assert scene2.character_by_name("OffChar") is restored_char
            assert scene2.location_by_name("OffLoc") is restored_loc
