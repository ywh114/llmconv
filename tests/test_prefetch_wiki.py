"""Tests for summarizer-generated wiki prefetch and orchestrator injection."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.models import GameRole, StreamResult
from ara.memory.chroma import ChromaStore
from ara.world.orchestrator import Orchestrator
from ara.world.story import Story
from ara.world.summarizer import Summarizer
from tests.test_items import _make_char, _make_scene


def _make_next_round_result(next_char: str, response_mode: str = "outer") -> StreamResult:
    return StreamResult(
        content="",
        tool_calls=[{
            "id": "call_next",
            "type": "function",
            "function": {
                "name": "next_round",
                "arguments": json.dumps({
                    "next_character": next_char,
                    "directive": "",
                    "suggestions": [],
                    "enter_characters": [],
                    "exit_characters": [],
                    "switch_location": "",
                    "edit_location": "",
                    "end_scene": False,
                    "next_scene": "",
                    "response_mode": response_mode,
                }),
            },
        }],
    )


def _make_story(tmp_path, db=None):
    config = MagicMock()
    config.data_dir = tmp_path / "data"
    config.language = "English"
    (tmp_path / "scene.toml").touch()
    return Story(config, db, MagicMock(spec=LLMClient), tmp_path / "scene.toml")


class TestSummarizerWikiPrefetch:
    """Tests for Summarizer.prefetch_wiki_context."""

    def test_prefetch_wiki_context_queries_wiki(self) -> None:
        class KeywordClient:
            def complete_subagent(self, task, context, max_tokens=512):
                return "ancient prophecy\nvampire council\nblood magic"

        summarizer = Summarizer(KeywordClient())  # type: ignore[arg-type]

        def recall_fn(
            query: str,
            exclude_docs: set[str] | None = None,
            max_distance: float | None = None,
        ) -> str:
            return f"- Result for {query}: lore"

        context = summarizer.prefetch_wiki_context(
            plot="The council meets to discuss the prophecy.",
            considerations="Foreshadow blood magic.",
            world="gothic_city",
            zeitgeist="dark fantasy",
            tone="tense",
            language="English",
            wiki_recall_fn=recall_fn,
        )

        assert "Result for ancient prophecy" in context
        assert "Result for vampire council" in context
        assert "Result for blood magic" in context

    def test_prefetch_wiki_context_skips_empty_results(self) -> None:
        class KeywordClient:
            def complete_subagent(self, task, context, max_tokens=512):
                return "ancient prophecy"

        summarizer = Summarizer(KeywordClient())  # type: ignore[arg-type]

        context = summarizer.prefetch_wiki_context(
            plot="The council meets.",
            considerations="",
            world="",
            zeitgeist="",
            tone="",
            language="English",
            wiki_recall_fn=lambda _q, exclude_docs=None, max_distance=None: "No relevant wiki entries found.",
        )

        assert context == ""


class TestStoryWikiPresence:
    """Tests for conditional wiki prefetch in Story."""

    def test_wiki_has_content_checks_collection_count(self, tmp_path) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.collection.return_value.count.return_value = 0
        story = _make_story(tmp_path, db=mock_db)
        assert not story._wiki_has_content()

        mock_db.collection.return_value.count.return_value = 3
        assert story._wiki_has_content()

    def test_wiki_has_content_false_when_db_missing(self, tmp_path) -> None:
        story = _make_story(tmp_path, db=None)
        assert not story._wiki_has_content()

    def test_initial_scene_prefetches_wiki_when_present(self, tmp_path) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.collection.return_value.count.return_value = 2
        story = _make_story(tmp_path, db=mock_db)
        story._current_path = tmp_path / "scene.toml"

        scene = MagicMock()
        scene.id = "start"
        scene.name = "Start"
        scene.scene_type = "scene"
        scene.plot_story = "The protagonist arrives."
        scene.plot_considerations = "Introduce the guild."
        scene.world = ""
        scene.zeitgeist = "test"
        scene.tone = "neutral"
        scene.language = "English"
        scene.time = "day"
        scene.settings = []
        scene.character_pool = []
        scene.location_pool = []
        scene.starting_characters = []
        scene.starting_location = MagicMock()
        scene.starting_location.name = "room"
        scene.player = None
        scene.narrator = None
        scene.next_choices = {}

        with patch("ara.world.story.Scene.load", return_value=scene):
            story._load_scene()

        assert story._summarizer.client.complete_subagent.called
        assert story.engine.orchestrator.prefetched_wiki == ""

    def test_initial_scene_skips_prefetch_when_wiki_empty(self, tmp_path) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.collection.return_value.count.return_value = 0
        story = _make_story(tmp_path, db=mock_db)
        story._current_path = tmp_path / "scene.toml"

        scene = MagicMock()
        scene.id = "start"
        scene.name = "Start"
        scene.scene_type = "scene"
        scene.plot_story = "The protagonist arrives."
        scene.plot_considerations = ""
        scene.world = ""
        scene.zeitgeist = "test"
        scene.tone = "neutral"
        scene.language = "English"
        scene.time = "day"
        scene.settings = []
        scene.character_pool = []
        scene.location_pool = []
        scene.starting_characters = []
        scene.starting_location = MagicMock()
        scene.starting_location.name = "room"
        scene.player = None
        scene.narrator = None
        scene.next_choices = {}

        with patch("ara.world.story.Scene.load", return_value=scene):
            story._load_scene()

        assert not story._summarizer.client.complete_subagent.called


class TestOrchestratorPrefetchWiki:
    """Tests for Orchestrator.prefetch_wiki and prompt injection."""

    def test_prefetch_wiki_annotates_trust(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"topic": "t1", "trust": 0.75}]],
        }
        client = MagicMock(spec=LLMClient)
        orch = Orchestrator(client, db=mock_db)

        result = orch.prefetch_wiki("query")

        assert "(trust: 0.75)" in result
        assert "doc1" in result

    def test_orchestrator_uses_prefetched_wiki(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        client = MagicMock()
        client.complete.return_value = _make_next_round_result("Alice")

        orch = Orchestrator(client, db=mock_db)
        orch.prefetched_wiki = "Prefetched lore about vampires."

        scene = _make_scene("prefetch_scene", mock_db)
        char = next(c for c in scene.character_pool if c.name == "Alice")
        ctx = ConversationContext("Player", "Narrator", "Alice")

        decision = orch.decide_next_turn(
            scene=scene,
            ctx=ctx,
            here_chars=scene.starting_characters,
            away_chars=set(),
            prev_char=None,
            loc=scene.starting_location,
        )

        assert decision.next_char == char

        calls = [c for c in client.complete.call_args_list
                 if c.kwargs.get("role") == GameRole.ORCHESTRATOR]
        assert calls
        call = calls[0]
        messages = call.kwargs.get("messages", [])
        joined = "\n".join(str(m.get("content", "")) for m in messages)
        assert "Prefetched lore about vampires." in joined

    def test_orchestrator_decision_log_includes_response_mode_and_attempts(
        self, caplog
    ) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        client = MagicMock()
        client.complete.return_value = _make_next_round_result("Alice", response_mode="outer_and_inner")

        orch = Orchestrator(client, db=mock_db)
        scene = _make_scene("log_scene", mock_db)
        ctx = ConversationContext("Player", "Narrator", "Alice")
        attempts = [
            {"source": "Player", "action": "sneak past the guard"},
            {"source": "Bob", "action": "create a distraction"},
        ]

        with caplog.at_level("INFO", logger="ara.world.orchestrator"):
            orch.decide_next_turn(
                scene=scene,
                ctx=ctx,
                here_chars=scene.starting_characters,
                away_chars=set(),
                prev_char=None,
                loc=scene.starting_location,
                attempts_for_orchestrator=attempts,
            )

        info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
        decision_msg = "\n".join(info_messages)
        assert "response_mode=outer_and_inner" in decision_msg
        assert "attempts=2" in decision_msg
        assert "Player: sneak past the guard" in decision_msg
        assert "Bob: create a distraction" in decision_msg
