"""Tests for the Story state machine and transition features."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.models import StreamResult
from ara.memory.chroma import ChromaStore
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Scene, SceneChoice
from ara.world.story import Story, _finalize_character
from ara.world.summarizer import Summarizer, TransitionRequest

from tests.helpers import ScriptedLLMClient as MockLLMClient
from tests.helpers import make_scene


def _make_scene(scene_id: str, next_choices: dict, mock_db: ChromaStore, char_names: list[str]) -> Scene:
    return make_scene(
        scene_id,
        mock_db,
        char_names=tuple(char_names),
        next_choices=next_choices,
        location_name="lab",
    )


def test_finalize_turn_emitted_on_location_change() -> None:
    """A finalize_turn event is emitted when the summarizer changes location."""
    mock_db = MagicMock(spec=ChromaStore)
    scene_a = _make_scene("scene_a", {"scene_b": SceneChoice(id="scene_b", desc="Next")}, mock_db, ["Player", "Alice", "Narrator"])
    scene_b = _make_scene("scene_b", {}, mock_db, ["Player", "Alice", "Narrator"])
    scene_b.starting_location.desc = "A burned lab."

    class MockClient:
        def complete(self, **kwargs):
            return StreamResult(content="Smoke still rises from the ruined equipment.")

        def complete_subagent(self, **kwargs):
            return "[summary]"

    mock_client = MockClient()
    engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.side_effect = [
        TurnDecision(next_char=scene_a.narrator, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=["Go"], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene="scene_b"),
        TurnDecision(next_char=scene_b.narrator, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        plot = Path(tmpdir) / "plot"
        plot.mkdir()
        (plot / "scene_a.toml").write_text('id = "scene_a"\n')
        (plot / "scene_b.toml").write_text('id = "scene_b"\n')

        config = AraSettings(data_dir=Path(tmpdir), api_key="", api_endpoint="", api_model="")
        story = Story(config, mock_db, mock_client, plot / "scene_a.toml")
        story.engine = engine

        scenes = {"scene_a": scene_a, "scene_b": scene_b}

        def fake_load(path, db, config, scene_history=None, **kwargs):
            return scenes[path.stem]

        with patch("ara.world.story.Scene.load", side_effect=fake_load):
            story.start()
            assert story.step().event == "scene_loaded"
            assert story.step().event == "turn"
            assert story.step().event == "needs_player_input"
            story.submit_player_input("Go")
            assert story.step().event == "transition"
            # The next step finalizes and loads the scene; we then step once more to get finalize_turn.
            story.step()  # scene_b loaded
            result = story.step()
            assert result.event == "finalize_turn"
            assert result.speaker == "Narrator"
            assert "Smoke" in result.output



def _make_scene_for_smoke(mock_db: ChromaStore) -> Scene:
    return make_scene("test", mock_db)


def test_state_machine_story_with_next_choice() -> None:
    """Drive a story with an actual next-scene choice (CLI smoke test)."""
    mock_db = MagicMock(spec=ChromaStore)
    scene = _make_scene_for_smoke(mock_db)
    scene.next_choices = {"tea": MagicMock(id="tea")}

    engine_decisions = [
        TurnDecision(
            next_char=scene.narrator,
            directive="Introduce",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene=None,
        ),
        TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=["Go to tea"],
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
            next_scene="tea",
        ),
    ]

    class MockClient:
        def complete(self, **kwargs):
            return StreamResult(content="mock")

        def complete_subagent(self, **kwargs):
            return "[sub-agent summary]"

    mock = MockClient()
    engine = Engine(mock)  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.side_effect = lambda **kw: engine_decisions.pop(0)

    story = Story(AraSettings(), mock_db, mock, Path(__file__))
    story.engine = engine

    with patch("ara.world.story.Scene.load", return_value=scene):
        story.start()
        assert not story.finished
        assert story._state == "loading"

        # Tick 1: load scene
        result = story.step()
        assert result.event == "scene_loaded", f"expected scene_loaded, got {result.event}"
        assert result.scene is scene

        # Tick 2: narrator turn
        result = story.step()
        assert result.event == "turn", f"expected turn, got {result.event}"
        assert engine.last_decision.next_char == scene.narrator

        # Tick 3: player turn requested
        result = story.step()
        assert result.event == "needs_player_input", f"expected needs_player_input, got {result.event}"
        assert result.suggestions == ["Go to tea"]

        # Provide player input
        story.submit_player_input("Let's go")
        assert not engine.needs_player_input

        # Tick 4: scene ends
        result = story.step()
        assert result.event == "transition", f"expected transition, got {result.event}"
        assert result.phase == "ended", f"expected phase ended, got {result.phase}"
        assert result.next_scene == "tea"

        # Next tick finalises and tries to load "tea.toml" which doesn't exist -> story_complete
        result = story.step()
        assert result.event == "story_complete", f"expected story_complete, got {result.event}"

        assert story.finished


def test_scratchpad_archived_and_carried_between_scenes() -> None:
    """End-of-scene finalization archives scratch; the next scene sees the archive."""
    mock_db = MagicMock(spec=ChromaStore)
    scene_a = _make_scene("scene_a", {"scene_b": SceneChoice(id="scene_b", desc="Next")}, mock_db, ["Player", "Alice", "Narrator"])
    scene_b = _make_scene("scene_b", {}, mock_db, ["Player", "Alice", "Narrator"])

    alice_a = next(c for c in scene_a.character_pool if c.name == "Alice")
    alice_a.scratch.text = "[Thought]: I should look for the key."

    responses = [
        # Narrator turn in scene_a.
        StreamResult(content="Welcome to the lab."),
        # _finalize_character for Alice: no tool call, scratch stays as-is.
        StreamResult(content="", tool_calls=[]),
        # Summarizer: produce a bridging summary for Alice.
        StreamResult(content="SUMMARY Alice: She remembers the search.\n\nLOCATION:\nA lab.\n\nTIME:\nmorning"),
    ]
    mock_client = MockLLMClient(responses)

    engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.side_effect = [
        TurnDecision(next_char=scene_a.narrator, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=["Go"], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene="scene_b"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        plot = Path(tmpdir) / "plot"
        plot.mkdir()
        (plot / "scene_a.toml").write_text('id = "scene_a"\n')
        (plot / "scene_b.toml").write_text('id = "scene_b"\n')

        config = AraSettings(data_dir=Path(tmpdir), api_key="", api_endpoint="", api_model="")
        story = Story(config, mock_db, mock_client, plot / "scene_a.toml")
        story.engine = engine

        scenes = {"scene_a": scene_a, "scene_b": scene_b}

        def fake_load(path, db, config, scene_history=None, **kwargs):
            return scenes[path.stem]

        with patch("ara.world.story.Scene.load", side_effect=fake_load):
            story.start()
            story.step()  # scene_loaded
            story.step()  # narrator turn
            story.step()  # needs_player_input
            story.submit_player_input("Go")
            story.step()  # transition
            story.step()  # finalize + load scene_b

    # After finalization, the old Alice's scratch should be archived.
    assert alice_a.scratch.text == "Nothing yet!"
    assert alice_a.scratch.prev_text == "[Thought]: I should look for the key."

    # The new Alice should have inherited the archived scratch.
    alice_b = next(c for c in scene_b.character_pool if c.name == "Alice")
    assert alice_b.scratch.prev_text == "[Thought]: I should look for the key."
    assert alice_b.scratch.text == "Nothing yet!"


def test_system_state_persists_across_scene_transition() -> None:
    """Player player status gained in scene A is present in scene B."""
    mock_db = MagicMock(spec=ChromaStore)
    scene_a = _make_scene("scene_a", {"scene_b": SceneChoice(id="scene_b", desc="Next")}, mock_db, ["Player", "Alice", "Narrator"])
    scene_b = _make_scene("scene_b", {}, mock_db, ["Player", "Alice", "Narrator"])

    responses = [
        StreamResult(content="Welcome to the lab."),
        StreamResult(content="Welcome again."),
        StreamResult(content="", tool_calls=[]),
        StreamResult(
            content="SUMMARY Alice: She remembers the search.\n\n"
            "LOCATION:\nA lab.\n\n"
            "TIME:\nmorning"
        ),
    ]
    mock_client = MockLLMClient(responses)

    engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.side_effect = [
        TurnDecision(next_char=scene_a.narrator, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=["Take key"], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None, system_changes={"inventory": ["Key"]}),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene="scene_b"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        plot = Path(tmpdir) / "plot"
        plot.mkdir()
        (plot / "scene_a.toml").write_text('id = "scene_a"\n')
        (plot / "scene_b.toml").write_text('id = "scene_b"\n')

        config = AraSettings(data_dir=Path(tmpdir), api_key="", api_endpoint="", api_model="")
        story = Story(config, mock_db, mock_client, plot / "scene_a.toml")
        story.engine = engine

        scenes = {"scene_a": scene_a, "scene_b": scene_b}

        def fake_load(path, db, config, scene_history=None, **kwargs):
            return scenes[path.stem]

        with patch("ara.world.story.Scene.load", side_effect=fake_load):
            story.start()
            story.step()  # scene_loaded
            story.step()  # narrator turn
            story.step()  # needs_player_input
            story.submit_player_input("Take key")
            story.step()  # applies system_changes, needs input again
            story.submit_player_input("Go")
            story.step()  # transition
            story.step()  # finalize + load scene_b

    assert story.engine.player_status == {
        "title": "Status",
        "sections": [{"type": "inventory", "items": ["Key"]}],
    }


def test_offscreen_character_status_persists() -> None:
    """A STATUS update for an off-screen character is applied when they return."""
    mock_db = MagicMock(spec=ChromaStore)
    scene_a = _make_scene("scene_a", {"scene_b": SceneChoice(id="scene_b", desc="Next")}, mock_db, ["Player", "Alice", "Narrator"])
    scene_b = _make_scene("scene_b", {}, mock_db, ["Player", "Alice", "Bob", "Narrator"])

    responses = [
        StreamResult(content="Welcome to the lab."),
        StreamResult(content="", tool_calls=[]),
        StreamResult(
            content="SUMMARY Alice: She is relieved.\n\n"
            "SUMMARY Bob: He limps in, clutching his side.\n\n"
            "STATUS Bob:\n{\"wounded\": true, \"location\": \"infirmary\"}\n\n"
            "LOCATION:\nA lab.\n\n"
            "TIME:\nmorning"
        ),
    ]
    mock_client = MockLLMClient(responses)

    engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.side_effect = [
        TurnDecision(next_char=scene_a.narrator, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=["Go"], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene="scene_b"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        plot = Path(tmpdir) / "plot"
        plot.mkdir()
        (plot / "scene_a.toml").write_text('id = "scene_a"\n')
        (plot / "scene_b.toml").write_text('id = "scene_b"\n')

        config = AraSettings(data_dir=Path(tmpdir), api_key="", api_endpoint="", api_model="")
        story = Story(config, mock_db, mock_client, plot / "scene_a.toml")
        story.engine = engine

        scenes = {"scene_a": scene_a, "scene_b": scene_b}

        def fake_load(path, db, config, scene_history=None, **kwargs):
            return scenes[path.stem]

        with patch("ara.world.story.Scene.load", side_effect=fake_load):
            story.start()
            story.step()  # scene_loaded
            story.step()  # narrator turn
            story.step()  # needs_player_input
            story.submit_player_input("Go")
            story.step()  # transition
            story.step()  # finalize + load scene_b

    bob_b = next(c for c in scene_b.character_pool if c.name == "Bob")
    assert bob_b.status == {"wounded": True, "location": "infirmary"}
    assert story._character_status["Bob"] == {"wounded": True, "location": "infirmary"}


def test_story_query_characters_searches_cards() -> None:
    """Story._query_characters searches character names and card text."""
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        story_name = "query_test"
        cc = tmp / "assets" / "cc" / story_name
        cc.mkdir(parents=True)
        (cc / "Alice").mkdir()
        (cc / "Alice" / "card.toml").write_text('name = "Alice"\nsummary = "A curious mechanic."\n')
        (cc / "Bob").mkdir()
        (cc / "Bob" / "card.toml").write_text('name = "Bob"\nsummary = "A grumpy guard."\n')
        (cc / "Eve").mkdir()

        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        scene_path = tmp / "assets" / "plot" / story_name / "scene1.toml"
        scene_path.parent.mkdir(parents=True)
        scene_path.write_text('id = "scene1"\n')
        story = Story(config, mock_db, mock_client, scene_path)

        results = story._query_characters("mechanic")
        assert any("Alice" in r for r in results)
        assert not any("Bob" in r for r in results)

        results = story._query_characters("Bob")
        assert any("Bob" in r for r in results)


def test_engine_records_mechanical_changelog() -> None:
    """Engine.step() should append mechanical changes to the changelog."""
    scene = _make_scene_for_smoke(MagicMock(spec=ChromaStore))

    mock = MockLLMClient([StreamResult(content="The room is quiet.")])
    engine = Engine(mock)  # type: ignore[arg-type]
    engine.orchestrator = MagicMock()
    engine.orchestrator.decide_next_turn.return_value = TurnDecision(
        next_char=scene.narrator,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        edit_location="The table was shattered into splinters.",
        set_time="evening",
        change_sprite={"NPC": "happy"},
        system_changes={"bars": {"HP": 90}},
        next_scene=None,
    )

    engine.start(scene)
    engine.step()

    assert engine.player_status == {
        "title": "Status",
        "sections": [{"type": "bars", "items": [{"label": "HP", "value": 90, "max": 100}]}],
    }
    assert engine.world_time == "evening"

    types = [entry["type"] for entry in engine.mechanical_changelog]
    assert "edit_location" in types
    assert "set_time" in types
    assert "change_sprite" in types
    assert "system_changes" in types

    system_entry = next(e for e in engine.mechanical_changelog if e["type"] == "system_changes")
    assert system_entry["changes"] == {"bars": {"HP": 90}}


class TestEngineLoop:
    """Integration tests of the conversation engine with a mock LLM."""

    def test_scene_ends_on_next_scene(self) -> None:
        """The engine should exit when the orchestrator returns a next_scene."""
        scene = _make_scene_for_smoke(MagicMock(spec=ChromaStore))

        orchestrator_result = StreamResult(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {
                        "name": "next_round",
                        "arguments": json.dumps({
                            "next_character": "Player",
                            "directive": "",
                            "suggestions": [],
                            "enter_characters": [],
                            "exit_characters": [],
                            "switch_location": "",
                            "next_scene": "end",
                        }),
                    },
                }
            ],
        )

        mock = MockLLMClient([orchestrator_result])
        engine = Engine(mock)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene="end",
        )

        inputs = ["hello"]
        def fake_input(prompt: str, suggestions: list[str]) -> str:
            return inputs.pop(0)

        result = engine.run(scene, get_user_input=fake_input)
        assert result == "end"

    def test_edit_location_updates_description(self) -> None:
        """The engine should update the location description when edit_location is set."""
        scene = _make_scene_for_smoke(MagicMock(spec=ChromaStore))
        original_desc = scene.starting_location.desc

        mock = MockLLMClient([StreamResult(content="The room is quiet.")])
        engine = Engine(mock)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=scene.narrator,
            directive="Describe the broken table",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            edit_location="The table was shattered into splinters.",
            next_scene=None,
        )

        engine.start(scene)
        engine.step()

        assert "shattered into splinters" in scene.starting_location.desc
        assert original_desc in scene.starting_location.desc


def test_finalize_character_uses_curated_view_and_sees_scratch() -> None:
    """Character finalizers see their own scratch and a single-assistant view of the scene."""
    mock_db = MagicMock(spec=ChromaStore)
    scene = _make_scene("scene", {}, mock_db, ["Player", "Alice", "Narrator"])
    alice = next(c for c in scene.character_pool if c.name == "Alice")
    alice.scratch.text = "[Journal]: my secret plan"

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def complete(self, **kwargs: Any) -> StreamResult:
            self.calls.append(kwargs)
            return StreamResult(content="", tool_calls=[])

        def complete_subagent(self, **kwargs: Any) -> str:
            return ""

    client = FakeClient()
    engine = Engine(client, db=mock_db)  # type: ignore[arg-type]
    engine._ctx = ConversationContext("Player", "Alice", "Bob", "Narrator")
    engine.ctx.enter_entities("Player", "Alice", "Bob", "Narrator")
    engine.ctx.user_message("Hi", name="Player", canonical_name="Player")
    engine.ctx.assistant_message("Hello", tool_calls=[], name="Bob", canonical_name="Bob")
    engine.ctx.user_message("Alice?", name="Player", canonical_name="Player")

    _finalize_character(alice, scene, engine, {})

    assert client.calls, "finalize_character should call the LLM"
    messages = client.calls[0]["messages"]

    # The character's own scratch is visible as an assistant turn.
    assert any(
        m.get("role") == "assistant" and "[Journal]: my secret plan" in (m.get("content") or "")
        for m in messages
    )

    # Other speakers appear as user messages in the curated view.
    assert any(
        m.get("role") == "user" and "Bob says: Hello" in (m.get("content") or "")
        for m in messages
    )

    # Bob must not remain an assistant in Alice's view.
    assert not any(
        m.get("role") == "assistant" and "Bob" in (m.get("name") or "")
        for m in messages
    )


def test_summarizer_receives_orchestrator_view_and_orchestrator_scratch() -> None:
    """The summarizer sees the orchestrator's curated view and all scratchpads."""
    mock_db = MagicMock(spec=ChromaStore)
    scene_a = _make_scene(
        "scene_a", {"scene_b": SceneChoice(id="scene_b", desc="Next")}, mock_db,
        ["Player", "Alice", "Narrator"]
    )
    scene_b = _make_scene("scene_b", {}, mock_db, ["Player", "Alice", "Narrator"])

    alice = next(c for c in scene_a.character_pool if c.name == "Alice")
    alice.scratch.text = "[Alice scratch]"

    responses = [
        StreamResult(content="Welcome to the lab."),
        StreamResult(content="", tool_calls=[]),
        StreamResult(
            content="SUMMARY Alice: She remembers.\n\nLOCATION:\nA lab.\n\nTIME:\nmorning"
        ),
    ]
    mock_client = MockLLMClient(responses)

    engine = Engine(mock_client, db=mock_db)  # type: ignore[arg-type]
    engine.orchestrator.scratch.text = "[Orchestrator scratch]"
    engine.orchestrator.decide_next_turn = MagicMock(side_effect=[
        TurnDecision(next_char=scene_a.narrator, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=["Go"], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene=None),
        TurnDecision(next_char=scene_a.player, directive="", suggestions=[], entering_chars=set(), exiting_chars=set(), switch_location=None, next_scene="scene_b"),
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        plot = Path(tmpdir) / "plot"
        plot.mkdir()
        (plot / "scene_a.toml").write_text('id = "scene_a"\n')
        (plot / "scene_b.toml").write_text('id = "scene_b"\n')

        config = AraSettings(data_dir=Path(tmpdir), api_key="", api_endpoint="", api_model="")
        story = Story(config, mock_db, mock_client, plot / "scene_a.toml")
        story.engine = engine

        scenes = {"scene_a": scene_a, "scene_b": scene_b}

        def fake_load(path: Path, db: ChromaStore, config: AraSettings, scene_history: list[str] | None = None, **kwargs: Any):
            return scenes[path.stem]

        captured: dict[str, Any] = {}
        original_summarize = Summarizer.summarize_transition

        def fake_summarize(self: Summarizer, request: TransitionRequest) -> Any:
            captured["conversation_context"] = request.conversation_context
            captured["scratchpads"] = request.scratchpads
            return original_summarize(self, request)

        with patch("ara.world.story.Scene.load", side_effect=fake_load), \
             patch("ara.world.story.Summarizer.summarize_transition", fake_summarize):
            story.start()
            story.step()  # scene_loaded
            story.step()  # narrator turn
            story.step()  # needs_player_input
            story.submit_player_input("Go")
            story.step()  # transition
            story.step()  # finalize + load scene_b

    assert "conversation_context" in captured
    assert "scratchpads" in captured

    # The summarizer receives the final in-scene scratchpads, including both
    # character scratches and the orchestrator journal.
    assert captured["scratchpads"].get("Alice") == "[Alice scratch]"
    assert captured["scratchpads"].get("Orchestrator") not in (None, "Nothing yet!")

    # The conversation context should be the orchestrator's non-collapsing curated view,
    # not the raw engine context.  In that view the orchestrator's own tool-call turns
    # remain assistant messages, while other speakers are user messages.
    conv_ctx = captured["conversation_context"]
    roles = [m.get("role") for m in conv_ctx]
    assert "assistant" in roles
    assert "user" in roles
    # Internal visibility markers must have been stripped.
    for msg in conv_ctx:
        assert not any(k.startswith("_") for k in msg.keys())
