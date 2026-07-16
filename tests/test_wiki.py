"""Unit tests for :class:`ara.memory.wiki.WikiStore`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from ara.memory.chroma import ChromaStore
from ara.memory.wiki import WIKI_COLLECTION, WikiStore

from tests.helpers import make_char


def _mock_db(
    documents: list[str] | None = None,
    metadatas: list[dict] | None = None,
    distances: list[float] | None = None,
    ids: list[str] | None = None,
) -> MagicMock:
    """Build a mock ChromaStore whose query() returns one result group."""
    mock_db = MagicMock(spec=ChromaStore)
    docs = documents if documents is not None else []
    result: dict = {"documents": [docs]}
    if ids is not None:
        result["ids"] = [ids]
    if metadatas is not None:
        result["metadatas"] = [metadatas]
    if distances is not None:
        result["distances"] = [distances]
    mock_db.query.return_value = result
    return mock_db


class TestNormalizeDoc:
    """normalize_doc produces a canonical form for deduplication."""

    def test_strips_dash_and_whitespace(self) -> None:
        assert WikiStore.normalize_doc("-  Some   fact. ") == "Some fact."

    def test_strips_trust_prefix(self) -> None:
        assert WikiStore.normalize_doc("(trust: 0.5) Some fact.") == "Some fact."

    def test_strips_dash_and_trust_together(self) -> None:
        assert WikiStore.normalize_doc("- (trust: -1.0)  A lie. ") == "A lie."


class TestRecall:
    """WikiStore.recall querying, filtering, and failure paths."""

    def test_no_db_returns_unavailable(self) -> None:
        assert WikiStore(None).recall("q") == "Wiki is not available."

    def test_empty_results(self) -> None:
        wiki = WikiStore(_mock_db([]))
        assert wiki.recall("q") == "No relevant wiki entries found."

    def test_formats_results_as_bullets(self) -> None:
        wiki = WikiStore(_mock_db(["Fact one.", "Fact two."]))
        assert wiki.recall("q") == "- Fact one.\n\n- Fact two."

    def test_max_distance_filters_far_documents(self) -> None:
        wiki = WikiStore(_mock_db(["Near.", "Far."], distances=[0.3, 0.9]))
        result = wiki.recall("q", max_distance=0.65)
        assert "Near." in result
        assert "Far." not in result

    def test_max_distance_all_filtered_returns_empty_message(self) -> None:
        wiki = WikiStore(_mock_db(["Far."], distances=[0.9]))
        assert wiki.recall("q", max_distance=0.65) == "No relevant wiki entries found."

    def test_max_distance_none_disables_filtering(self) -> None:
        wiki = WikiStore(_mock_db(["Far."], distances=[0.9]))
        assert "Far." in wiki.recall("q", max_distance=None)

    def test_exclude_docs_filters_normalized_duplicates(self) -> None:
        wiki = WikiStore(_mock_db(["Keep this.", "Drop this."]))
        excluded = {WikiStore.normalize_doc("- (trust: 0.1) Drop this.")}
        result = wiki.recall("q", exclude_docs=excluded)
        assert "Keep this." in result
        assert "Drop this." not in result

    def test_exclude_all_returns_covered_message(self) -> None:
        wiki = WikiStore(_mock_db(["Drop this."]))
        excluded = {WikiStore.normalize_doc("Drop this.")}
        result = wiki.recall("q", exclude_docs=excluded)
        assert result == "All results already covered by existing context."

    def test_dedup_against_prefetched_text(self) -> None:
        wiki = WikiStore(_mock_db(["New lore.", "Already prefetched."]))
        result = wiki.recall("q", dedup_against="- Already prefetched.")
        assert "New lore." in result
        assert "Already prefetched." not in result

    def test_annotate_trust_prefixes_scores(self) -> None:
        wiki = WikiStore(_mock_db(
            ["Canon.", "Rumor.", "Untrusted."],
            metadatas=[{"trust": 0.8}, {"trust": -1.0}, {}],
        ))
        result = wiki.recall("q", annotate_trust=True)
        assert result == "- (trust: 0.8) Canon.\n\n- (trust: -1.0) Rumor.\n\n- Untrusted."

    def test_db_error_returns_failure_message(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.query.side_effect = RuntimeError("boom")
        assert WikiStore(mock_db).recall("q") == "Wiki recall failed."


class TestWrite:
    """WikiStore.write upserts entries with trust metadata."""

    def test_no_db_returns_unavailable(self) -> None:
        assert WikiStore(None).write("t", "c") == "Wiki is not available."

    def test_write_upserts_with_metadata(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        result = WikiStore(mock_db).write("topic", "content", importance="important", trust=0.8)
        assert result == "Wiki entry 'topic' saved."
        mock_db.upsert.assert_called_once_with(
            WIKI_COLLECTION,
            ids=["topic"],
            documents=["content"],
            metadatas=[{"topic": "topic", "importance": "important", "trust": 0.8}],
        )

    def test_write_default_importance_and_trust(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        WikiStore(mock_db).write("topic", "content")
        meta = mock_db.upsert.call_args.kwargs["metadatas"][0]
        assert meta["importance"] == "notable"
        assert meta["trust"] == 0.0

    def test_db_error_returns_failure_message(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.upsert.side_effect = RuntimeError("boom")
        assert WikiStore(mock_db).write("t", "c") == "Wiki write failed."


class TestForget:
    """WikiStore.forget deletes entries from the collection."""

    def test_no_db_returns_unavailable(self) -> None:
        assert WikiStore(None).forget("t") == "Wiki is not available."

    def test_forget_deletes_by_id(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        result = WikiStore(mock_db).forget("topic")
        assert result == "Wiki entry 'topic' deleted."
        mock_db.collection.assert_called_once_with(WIKI_COLLECTION)
        mock_db.collection.return_value.delete.assert_called_once_with(ids=["topic"])

    def test_db_error_returns_failure_message(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.collection.side_effect = RuntimeError("boom")
        assert WikiStore(mock_db).forget("t") == "Wiki forget failed."


class TestHasContent:
    """WikiStore.has_content reflects collection count."""

    def test_no_db_is_false(self) -> None:
        assert WikiStore(None).has_content() is False

    def test_empty_collection_is_false(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.collection.return_value.count.return_value = 0
        assert WikiStore(mock_db).has_content() is False

    def test_non_empty_collection_is_true(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.collection.return_value.count.return_value = 3
        assert WikiStore(mock_db).has_content() is True


class TestFilterForQuerier:
    """The querier-aware subagent reframes raw documents."""

    def test_empty_docs_passthrough_without_client(self) -> None:
        client = MagicMock()
        wiki = WikiStore(MagicMock(spec=ChromaStore), client=client)
        assert wiki.filter_for_querier("q", [], [], [], MagicMock()) == []
        client.complete_subagent.assert_not_called()

    def test_context_includes_querier_and_source_metadata(self) -> None:
        client = MagicMock()
        client.complete_subagent.return_value = "A maritime federation."
        wiki = WikiStore(MagicMock(spec=ChromaStore), client=client)
        querier = make_char("Novice", MagicMock(spec=ChromaStore))

        docs = wiki.filter_for_querier(
            "Eagle Union",
            ["Eagle Union is a maritime federation."],
            ["id1"],
            [{"world": "azur_lane", "topic": "world:azur_lane:factions:Eagle Union"}],
            querier,
        )

        assert docs == ["A maritime federation."]
        context = client.complete_subagent.call_args.kwargs["context"]
        assert "Querier: Novice" in context
        assert "Novice personality" in context
        assert "world:azur_lane:factions:Eagle Union" in context
        assert "world: azur_lane" in context

    def test_blank_subagent_answer_falls_back_to_raw_docs(self) -> None:
        client = MagicMock()
        client.complete_subagent.return_value = "   "
        wiki = WikiStore(MagicMock(spec=ChromaStore), client=client)
        querier = make_char("Novice", MagicMock(spec=ChromaStore))

        docs = wiki.filter_for_querier("q", ["Raw doc."], ["id1"], [{}], querier)
        assert docs == ["Raw doc."]


class TestIngestNarrativeState:
    """Narrative state is mirrored into the wiki as a critical entry."""

    def test_no_db_or_empty_state_is_noop(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        WikiStore(None).ingest_narrative_state({"a": 1})
        WikiStore(mock_db).ingest_narrative_state({})
        mock_db.upsert.assert_not_called()

    def test_state_upserted_as_json(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        state = {"war_declared": True}
        WikiStore(mock_db).ingest_narrative_state(state)
        mock_db.upsert.assert_called_once_with(
            WIKI_COLLECTION,
            ids=["story:state"],
            documents=[json.dumps(state, ensure_ascii=False)],
            metadatas=[{"topic": "story:state", "importance": "critical", "trust": 1.0}],
        )


class TestIngestInventedFacts:
    """Summarizer-invented facts are persisted with source and trust."""

    def test_no_facts_or_no_db_is_noop(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        WikiStore(None).ingest_invented_facts([{"fact": "Something."}])
        WikiStore(mock_db).ingest_invented_facts([])
        mock_db.upsert.assert_not_called()

    def test_blank_statements_are_skipped(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        WikiStore(mock_db).ingest_invented_facts([{"fact": "  "}, {"trust": 0.5}])
        mock_db.upsert.assert_not_called()

    def test_fact_upserted_with_source_and_trust(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        WikiStore(mock_db).ingest_invented_facts(
            [{"fact": "The vase shattered.", "trust": 1.0, "source": "Narrator"}]
        )
        mock_db.upsert.assert_called_once_with(
            WIKI_COLLECTION,
            ids=["invented_fact_000"],
            documents=["The vase shattered.\nSource: Narrator"],
            metadatas=[{
                "topic": "invented_fact_000",
                "importance": "notable",
                "trust": 1.0,
            }],
        )


class TestIngestSettingFile:
    """World-setting TOML files are bulk-ingested into the wiki."""

    def _write_setting(self, tmp_path: Path) -> Path:
        path = tmp_path / "test_world.toml"
        path.write_text(
            'id = "test_world"\n'
            'name = "Test World"\n'
            'summary = "A world for tests."\n'
            '\n'
            '[[factions]]\n'
            'name = "Eagle Union"\n'
            'description = "A maritime federation."\n'
        )
        return path

    def test_missing_path_is_noop(self, tmp_path: Path) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        wiki = WikiStore(mock_db)
        wiki.ingest_setting_file(None)
        wiki.ingest_setting_file(tmp_path / "missing.toml")
        mock_db.upsert.assert_not_called()

    def test_setting_entries_upserted_with_world_metadata(self, tmp_path: Path) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        WikiStore(mock_db).ingest_setting_file(self._write_setting(tmp_path))

        mock_db.upsert.assert_called_once()
        call = mock_db.upsert.call_args
        assert call.args[0] == WIKI_COLLECTION
        ids = call.kwargs["ids"]
        assert "world:test_world:summary" in ids
        assert "world:test_world:factions:Eagle Union" in ids
        docs = dict(zip(ids, call.kwargs["documents"]))
        assert docs["world:test_world:summary"] == "A world for tests."
        assert "A maritime federation." in docs["world:test_world:factions:Eagle Union"]
        for meta in call.kwargs["metadatas"]:
            assert meta["importance"] == "critical"
            assert meta["world"] == "test_world"

    def test_invalid_toml_does_not_raise(self, tmp_path: Path) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        bad = tmp_path / "bad.toml"
        bad.write_text("not [valid toml")
        WikiStore(mock_db).ingest_setting_file(bad)
        mock_db.upsert.assert_not_called()

    def test_no_db_returns_before_upsert(self, tmp_path: Path) -> None:
        # Parsing still happens, but nothing is stored and nothing crashes.
        WikiStore(None).ingest_setting_file(self._write_setting(tmp_path))
