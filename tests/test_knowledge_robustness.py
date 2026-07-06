"""Robustness tests for the knowledge / memory system.

These tests document and guard against regressing the behavior of the
knowledge layer after the subagent prompts were changed to prohibit invention:

1. The subagent filter in both CharacterMemory.recall and
   Orchestrator._wiki_recall must answer ONLY from the provided raw
   memories/documents.  When nothing is relevant it returns
   ``-- nothing relevant found`` instead of inventing facts.

2. The subagent still collapses multiple raw memories/documents into a single
   synthesized paragraph, discarding multiplicity and provenance, but it must
   not hallucinate new information.

3. CharacterMemory only recalls documents whose metadata contain
   ``memory=True``; anything else is invisible.

4. Wiki recall defaults to ``trust=0.0`` (invented) on writes unless the caller
   provides a trust score, and trust annotations are textual prefixes that the
   subagent may reframe or ignore.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from ara.memory.knowledge import Scratchpad

from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, NullMemory
from ara.world.character import Character, Importance
from ara.world.orchestrator import Orchestrator


def _make_character(name: str = "Alice") -> Character:
    return Character(
        id=uuid.uuid5(uuid.NAMESPACE_DNS, f"test.{name}"),
        canonical_name=name,
        name=name,
        card_fields={
            "name": name,
            "summary": f"{name} is a test character.",
            "personality": "curious and chatty",
            "scenario": "investigating a mystery",
            "greeting_message": f"Hi, I'm {name}",
            "example_messages": "",
        },
        importance=Importance.IMPORTANT,
        memory=MagicMock(),  # placeholder, overridden in tests
        scratch=MagicMock(),
    )


class MockLLMClient:
    """Fake LLM client that returns whatever the test configures."""

    def __init__(self, subagent_answer: str = "") -> None:
        self.subagent_answer = subagent_answer
        self.subagent_calls: list[dict] = []

    def complete_subagent(
        self, task: str, context: str, system_prompt: str = "", max_tokens: int = 512
    ) -> str:
        self.subagent_calls.append({"task": task, "context": context})
        return self.subagent_answer


def _make_mock_db(
    *,
    documents: list[list[str]],
    metadatas: list[list[dict]] | None = None,
) -> MagicMock:
    """Build a mock ChromaStore whose query() returns the given docs/metas/ids."""
    mock_db = MagicMock(spec=ChromaStore)
    if metadatas is None:
        metadatas = [[{} for _ in group] for group in documents]
    ids = [
        [f"doc_{group_idx}:{idx}" for idx in range(len(group))]
        for group_idx, group in enumerate(documents)
    ]
    mock_db.query.return_value = {
        "documents": documents,
        "metadatas": metadatas,
        "ids": ids,
    }
    return mock_db


class TestCharacterMemoryRobustness:
    """Robustness checks for per-character memory."""

    def test_recall_returns_raw_memories_when_subagent_disabled(self) -> None:
        """Without a client/querier, recall returns the raw DB documents."""
        mock_db = _make_mock_db(documents=[["Alice likes tea.", "Alice hates coffee."]])
        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)

        result = mem.recall(["What does Alice drink?"])

        assert result == ["Alice likes tea.", "Alice hates coffee."]
        mock_db.query.assert_called_once()

    def test_recall_does_not_invent_when_memory_store_is_empty(self) -> None:
        """An empty memory store returns nothing even with the subagent filter.

        CharacterMemory.recall only invokes the subagent when ``flat`` is
        non-empty, so a completely blank store does not hallucinate memories.
        """
        mock_db = _make_mock_db(documents=[[]])
        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)
        client = MockLLMClient(subagent_answer="Alice once met a dragon.")
        querier = _make_character("Bob")

        result = mem.recall(["Tell me about Alice."], client=client, querier=querier)

        assert result == []
        assert len(client.subagent_calls) == 0

    def test_recall_admits_when_only_irrelevant_memories_exist_with_subagent(self) -> None:
        """With only irrelevant memories, the subagent admits nothing relevant.

        The DB returned *something* (so ``flat`` is non-empty), but the retrieved
        snippets do not answer the query.  The subagent must now return the
        admission marker rather than invent a dragon story.
        """
        mock_db = _make_mock_db(documents=[["Alice likes tea.", "Alice wears blue."]])
        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)
        client = MockLLMClient(subagent_answer="-- nothing relevant found")
        querier = _make_character("Bob")

        result = mem.recall(["Tell me about Alice's dragon."], client=client, querier=querier)

        assert result == ["-- nothing relevant found"]
        assert len(client.subagent_calls) == 1

    def test_recall_does_not_invent_without_subagent(self) -> None:
        """Raw recall is honest: no matches returns an empty list."""
        mock_db = _make_mock_db(documents=[[]])
        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)

        result = mem.recall(["Unknown query"])

        assert result == []

    def test_recall_collapses_multiple_memories_into_one_subagent_answer(self) -> None:
        """The subagent replaces many raw memories with one synthesized paragraph.

        This discards nuance and provenance.  A character with contradictory
        memories may receive a single smoothed-over summary instead of the
        original contradictory sources.
        """
        mock_db = _make_mock_db(
            documents=[[
                "Alice trusted Bob before the incident.",
                "Alice now fears Bob after the incident.",
            ]]
        )
        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)
        client = MockLLMClient(subagent_answer="Alice has complicated feelings about Bob.")
        querier = _make_character("Carol")

        result = mem.recall(["How does Alice feel about Bob?"], client=client, querier=querier)

        assert result == ["Alice has complicated feelings about Bob."]
        assert len(result) == 1

    def test_recall_respects_memory_metadata_filter(self) -> None:
        """Documents without ``memory=True`` are invisible to recall.

        This means any tool or module that writes to a character collection
        without setting the ``memory`` metadata effectively creates ghost data.
        """
        mock_db = _make_mock_db(
            documents=[["This is a memory.", "This is system lore."]],
            metadatas=[[{"memory": True}, {"memory": False}]],
        )
        # The mock returns both docs, but the real implementation passes
        # where={'memory': {'$eq': True}} to the query.  We verify the filter
        # is passed; a real ChromaDB would then drop the second doc.
        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)

        mem.recall(["anything"])

        call = mock_db.query.call_args
        assert call.kwargs["where"] == {"memory": {"$eq": True}}

    def test_recall_depth_changes_result_limit(self) -> None:
        """Different depths request different numbers of results from the DB."""
        mock_db = _make_mock_db(documents=[["a", "b", "c", "d", "e", "f"]])
        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)

        mem.recall(["x"], depth="shallow")
        assert mock_db.query.call_args.kwargs["n_results"] == 2

        mem.recall(["x"], depth="very_deep")
        assert mock_db.query.call_args.kwargs["n_results"] == 30

    def test_add_conversation_then_recall_roundtrip(self) -> None:
        """After adding memories, recall should retrieve them (raw mode)."""
        mock_db = MagicMock(spec=ChromaStore)
        # Simulate the DB echoing back whatever was upserted.
        stored: list[str] = []

        def fake_upsert(collection, ids, documents, metadatas):
            stored.extend(documents)

        def fake_query(collection, query_texts, n_results, where):
            return {"documents": [stored[:n_results]], "metadatas": [[{"memory": True}] * len(stored[:n_results])]}

        mock_db.upsert.side_effect = fake_upsert
        mock_db.query.side_effect = fake_query

        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)
        mem.add_conversation(["Alice learned the secret password.", "Alice forgot her umbrella."])

        result = mem.recall(["What does Alice know?"])
        assert "Alice learned the secret password." in result

    def test_recall_subagent_receives_querier_identity(self) -> None:
        """The subagent is told who is asking and their background."""
        mock_db = _make_mock_db(documents=[["Alice likes tea."]])
        mem = CharacterMemory(character_id=uuid.uuid4(), db=mock_db)
        client = MockLLMClient(subagent_answer="ok")
        querier = _make_character("Expert")
        querier.card_fields["personality"] = "seasoned detective"
        querier.card_fields["scenario"] = "interrogating a witness"

        mem.recall(["Alice"], client=client, querier=querier)

        context = client.subagent_calls[0]["context"]
        assert "Querier: Expert" in context
        assert "seasoned detective" in context
        assert "interrogating a witness" in context


class TestOrchestratorWikiRobustness:
    """Robustness checks for the orchestrator shared wiki."""

    def test_wiki_recall_returns_raw_documents(self) -> None:
        """Without querier/annotation, recall returns raw wiki documents."""
        mock_db = _make_mock_db(documents=[["Fact one.", "Fact two."]])
        orch = Orchestrator(MagicMock(spec=LLMClient), db=mock_db)

        result = orch._wiki_recall("query")

        assert "Fact one." in result
        assert "Fact two." in result

    def test_wiki_recall_annotates_trust_scores(self) -> None:
        """Trust annotations are rendered as textual prefixes."""
        mock_db = _make_mock_db(
            documents=[["Canon fact.", "Rumor fact."]],
            metadatas=[[{"trust": 1.0}, {"trust": 0.4}]],
        )
        orch = Orchestrator(MagicMock(spec=LLMClient), db=mock_db)

        result = orch._wiki_recall("query", annotate_trust=True)

        assert "(trust: 1.0) Canon fact." in result
        assert "(trust: 0.4) Rumor fact." in result

    def test_wiki_recall_handles_missing_trust_gracefully(self) -> None:
        """Documents without trust metadata do not crash annotation."""
        mock_db = _make_mock_db(documents=[["Untrusted fact."]])
        orch = Orchestrator(MagicMock(spec=LLMClient), db=mock_db)

        result = orch._wiki_recall("query", annotate_trust=True)

        assert "Untrusted fact." in result
        assert "(trust:" not in result

    def test_wiki_recall_admits_when_querier_filter_enabled(self) -> None:
        """The wiki subagent admits ignorance when raw documents don't match.

        Unlike the previous behavior, the querier-aware filter must now return
        the admission marker instead of manufacturing consistent-sounding lore.
        """
        mock_db = _make_mock_db(documents=[["Some unrelated lore."]])
        client = MockLLMClient(subagent_answer="-- nothing relevant found")
        orch = Orchestrator(client, db=mock_db)
        querier = _make_character("Novice")

        result = orch._wiki_recall("secret prophecy", querier=querier)

        assert "-- nothing relevant found" in result

    def test_wiki_recall_returns_empty_message_when_no_docs(self) -> None:
        """Empty wiki results produce a clear message instead of crashing."""
        mock_db = _make_mock_db(documents=[[]])
        orch = Orchestrator(MagicMock(spec=LLMClient), db=mock_db)

        result = orch._wiki_recall("unknown")

        assert result == "No relevant wiki entries found."

    def test_wiki_write_default_trust_is_invented(self) -> None:
        """The default trust on a wiki write is 0.0, i.e. 'invented'.

        Callers must explicitly pass a trust value; otherwise the entry is
        stored as fabricated by default.
        """
        mock_db = MagicMock(spec=ChromaStore)
        orch = Orchestrator(MagicMock(spec=LLMClient), db=mock_db)

        orch._wiki_write("topic", "content")

        meta = mock_db.upsert.call_args.kwargs["metadatas"][0]
        assert meta["trust"] == 0.0

    def test_wiki_write_negative_trust_persists(self) -> None:
        """Explicit lies can be stored with negative trust."""
        mock_db = MagicMock(spec=ChromaStore)
        orch = Orchestrator(MagicMock(spec=LLMClient), db=mock_db)

        orch._wiki_write("topic", "content", trust=-1.0)

        meta = mock_db.upsert.call_args.kwargs["metadatas"][0]
        assert meta["trust"] == -1.0

    def test_wiki_querier_filter_collapses_multiple_docs(self) -> None:
        """Querier filtering collapses the raw wiki list into one bullet.

        Synthesis from the provided documents is still allowed; invention is not.
        """
        mock_db = _make_mock_db(
            documents=[["Doc A contradicts doc B.", "Doc B contradicts doc A."]]
        )
        client = MockLLMClient(subagent_answer="The documents contradict each other.")
        orch = Orchestrator(client, db=mock_db)
        querier = _make_character("Scholar")

        result = orch._wiki_recall("contradiction", querier=querier)

        # Result is a single bullet containing the synthesized answer.
        assert result.count("-") == 1
        assert "The documents contradict each other." in result

    def test_wiki_querier_filter_receives_source_world_metadata(self) -> None:
        """The filtering subagent gets topic IDs and source-world metadata."""
        mock_db = _make_mock_db(
            documents=[["Eagle Union is a maritime federation."]],
            metadatas=[[{"world": "azur_lane", "topic": "world:azur_lane:factions:Eagle Union"}]],
        )
        client = MockLLMClient(subagent_answer="A maritime federation.")
        orch = Orchestrator(client, db=mock_db)
        querier = _make_character("Novice")

        orch._wiki_recall("Eagle Union", querier=querier)

        assert len(client.subagent_calls) == 1
        context = client.subagent_calls[0]["context"]
        assert "world:azur_lane:factions:Eagle Union" in context
        assert "world: azur_lane" in context


class TestNullMemory:
    """NullMemory should remain a safe no-op."""

    def test_null_memory_add_and_recall_are_noops(self) -> None:
        mem = NullMemory()
        mem.add_conversation(["anything"])
        assert mem.recall(["anything"]) == []


class TestScratchpad:
    """Tests for the ephemeral scratchpad."""

    def test_prepare_for_new_scene(self) -> None:
        """Preparing for a new scene should archive the current text."""
        pad = Scratchpad(text="Current plans")
        pad.prepare_for_new_scene()
        assert pad.prev_text == "Current plans"
        assert pad.text == "Nothing yet!"


class TestCharacterMemoryBasics:
    """Basic unit tests for vector-backed character memory."""

    def test_recall_flattening(self) -> None:
        """Recall should flatten nested document groups from ChromaDB."""
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.query.return_value = {
            "documents": [
                ["doc1", "doc2"],
                ["doc3"],
            ]
        }

        mem = CharacterMemory(
            character_id=uuid.uuid4(),
            db=mock_db,
        )
        results = mem.recall(["query"], depth="medium")

        assert results == ["doc1", "doc2", "doc3"]
        mock_db.query.assert_called_once()

    def test_add_conversation_skips_empty(self) -> None:
        """``add_conversation`` should be a no-op when given an empty list."""
        mock_db = MagicMock(spec=ChromaStore)
        mem = CharacterMemory(
            character_id=uuid.uuid4(),
            db=mock_db,
        )
        mem.add_conversation([])
        mock_db.upsert.assert_not_called()
