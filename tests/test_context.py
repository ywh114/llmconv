"""Tests for :class:`ara.llm.context.ConversationContext`."""

from __future__ import annotations

import pytest

from ara.llm.context import ConversationContext


class TestConversationContext:
    """Unit tests for slice-based entity visibility and message sequencing."""

    def test_enter_exit_visibility(self) -> None:
        """Entities should only see messages from rounds they were present."""
        ctx = ConversationContext("Alice", "Bob")
        ctx.enter_entities("Alice", "Bob")

        ctx.user_message("Hello", name="Alice")
        ctx.assistant_message("Hi there", tool_calls=[], name="Bob")

        ctx.exit_entities("Bob")
        ctx.user_message("Secret", name="Alice")
        ctx.assistant_message("Got it", tool_calls=[], name="Alice")

        alice_view = ctx.to_list("Alice")
        bob_view = ctx.to_list("Bob")

        # Alice sees all 4 messages
        assert len(alice_view) == 4
        # Bob only sees the first 2
        assert len(bob_view) == 2

    def test_branch_isolation(self) -> None:
        """Modifying a branch must not affect the base context."""
        ctx = ConversationContext("Alice")
        ctx.enter_entities("Alice")
        ctx.user_message("Base msg", name="Alice")
        ctx.assistant_message("Base reply", tool_calls=[], name="Alice")

        branch = ctx.branch()
        branch.user_message("Branch msg", name="Alice")

        assert len(ctx.to_list()) == 2
        assert len(branch.to_list()) == 3

    def test_user_auto_pads_after_user(self) -> None:
        """A user message following another user message auto-inserts padding."""
        ctx = ConversationContext("Alice")
        ctx.enter_entities("Alice")
        ctx.user_message("First", name="Alice")
        ctx.user_message("Second", name="Alice")
        assert len(ctx.to_list()) == 3  # user, assistant padding, user

    def test_assistant_auto_pads_after_assistant(self) -> None:
        """An assistant message following another assistant message auto-inserts padding."""
        ctx = ConversationContext("Alice")
        ctx.enter_entities("Alice")
        ctx.user_message("Question", name="Alice")
        ctx.assistant_message("Answer", tool_calls=[], name="Alice")
        ctx.assistant_message("Another", tool_calls=[], name="Alice")
        assert len(ctx.to_list()) == 4  # user, assistant, user padding, assistant

    def test_reasoning_content_preserved_without_tools(self) -> None:
        """Non-tool assistant messages retain ``reasoning_content`` to satisfy
        DeepSeek's requirement that reasoning content is never dropped."""
        ctx = ConversationContext("Alice")
        ctx.enter_entities("Alice")
        ctx.user_message("Q", name="Alice")
        ctx.assistant_message(
            "A", tool_calls=[], name="Alice", reasoning_content="thinking..."
        )
        msg = ctx.to_list("Alice")[-1]
        assert msg.get("reasoning_content") == "thinking..."

    def test_reasoning_content_preserved_with_tools(self) -> None:
        """Tool-call assistant messages must retain ``reasoning_content``."""
        ctx = ConversationContext("Alice")
        ctx.enter_entities("Alice")
        ctx.user_message("Q", name="Alice")
        ctx.assistant_message(
            "A",
            tool_calls=[
                {
                    "id": "call_1",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
            name="Alice",
            reasoning_content="thinking...",
        )
        msg = ctx.to_list("Alice")[-1]
        assert msg.get("reasoning_content") == "thinking..."

    def test_multiple_tool_results_after_assistant_tool_call(self) -> None:
        """A single assistant message with multiple tool calls can be followed by
        multiple tool-result messages in the same turn."""
        ctx = ConversationContext("Alice")
        ctx.enter_entities("Alice")
        ctx.user_message("Q", name="Alice")
        ctx.assistant_message(
            "",
            tool_calls=[
                {"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}},
                {"id": "call_2", "function": {"name": "lookup", "arguments": "{}"}},
            ],
            name="Alice",
        )
        ctx.tool_message("result 1", "call_1")
        ctx.tool_message("result 2", "call_2")

        messages = ctx.to_list("Alice")
        assert messages[-2]["role"] == "tool"
        assert messages[-2]["tool_call_id"] == "call_1"
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["tool_call_id"] == "call_2"

    def test_filter_normalizes_adjacent_assistants_after_hidden_user(self) -> None:
        """A hidden user message between two assistant replies must not leave
        two assistant messages in a row for observers."""
        ctx = ConversationContext("Host", "NPC")
        ctx.enter_entities("Host", "NPC")
        ctx.assistant_message("Host speaks", tool_calls=[], name="Host")
        ctx.user_message(
            "Player thought", name="Player", hidden=True, visible_to={"Host"}
        )
        ctx.assistant_message("Host replies", tool_calls=[], name="Host")

        roles = [m["role"] for m in ctx.to_list("NPC")]
        assert roles == ["assistant", "user", "assistant"]

    def test_filter_normalizes_adjacent_users_after_hidden_assistant(self) -> None:
        """A hidden assistant message between two user messages must not leave
        two user messages in a row for observers."""
        ctx = ConversationContext("Host", "NPC")
        ctx.enter_entities("Host", "NPC")
        ctx.user_message("NPC speaks", name="NPC")
        ctx.assistant_message(
            "Hidden line", tool_calls=[], name="Player", hidden=True, visible_to={"Host"}
        )
        ctx.user_message("NPC speaks again", name="NPC")

        roles = [m["role"] for m in ctx.to_list("NPC")]
        assert roles == ["user", "assistant", "user"]

    def test_filter_drops_orphaned_tool_results(self) -> None:
        """Tool results whose matching assistant was hidden must be dropped."""
        ctx = ConversationContext("Host", "NPC")
        ctx.enter_entities("Host", "NPC")
        ctx.user_message("Question", name="NPC")
        ctx.assistant_message(
            "",
            tool_calls=[
                {"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}
            ],
            name="Player",
            hidden=True,
            visible_to={"Host"},
        )
        ctx.tool_message("result", "call_1")

        roles = [m["role"] for m in ctx.to_list("NPC")]
        assert roles == ["user"]

    def test_filter_strips_pending_tool_calls_at_end(self) -> None:
        """An assistant message with unresolved tool calls at the end of a view
        must have those tool calls stripped."""
        ctx = ConversationContext("Alice")
        ctx.enter_entities("Alice")
        ctx.user_message("Question", name="Alice")
        ctx.assistant_message(
            "",
            tool_calls=[
                {"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}
            ],
            name="Alice",
        )

        messages = ctx.to_list("Alice")
        assert messages[-1]["role"] == "assistant"
        assert "tool_calls" not in messages[-1]

    # ------------------------------------------------------------------ #
    # Curated (single-assistant) view tests
    # ------------------------------------------------------------------ #

    def test_curated_view_keeps_observer_as_assistant(self) -> None:
        """The observer's own assistant turns stay as assistant messages."""
        ctx = ConversationContext("Alice", "Bob", "Player")
        ctx.enter_entities("Alice", "Bob", "Player")
        ctx.user_message("Hello", name="Player", canonical_name="Player")
        ctx.assistant_message("Hi", tool_calls=[], name="Bob", canonical_name="Bob")
        ctx.user_message("Alice, your turn", name="Player", canonical_name="Player")
        ctx.assistant_message("Hey", tool_calls=[], name="Alice", canonical_name="Alice")

        branch = ctx.branch()
        view = branch.curated_view("Alice")
        roles = [m["role"] for m in view]
        assert roles == ["user", "assistant"]
        assert view[0]["content"] == "Player says: Hello\nBob says: Hi\nPlayer says: Alice, your turn"
        assert view[1]["content"] == "Hey"

    def test_curated_view_preserves_observer_tool_calls(self) -> None:
        """Observer tool-call turns and their results keep their roles."""
        ctx = ConversationContext("Alice", "Player")
        ctx.enter_entities("Alice", "Player")
        ctx.user_message("Question", name="Player", canonical_name="Player")
        ctx.assistant_message(
            "",
            tool_calls=[
                {"id": "tc1", "function": {"name": "recall", "arguments": "{}"}}
            ],
            name="Alice",
            canonical_name="Alice",
        )
        ctx.tool_message("memory result", "tc1")
        ctx.assistant_message("Answer", tool_calls=[], name="Alice", canonical_name="Alice")

        branch = ctx.branch()
        view = branch.curated_view("Alice")
        roles = [m["role"] for m in view]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert view[1].get("tool_calls")
        assert view[2]["content"] == "memory result"

    def test_curated_view_reports_other_tool_calls(self) -> None:
        """Other speakers' tool calls are collapsed into the user block."""
        ctx = ConversationContext("Alice", "Bob", "Player")
        ctx.enter_entities("Alice", "Bob", "Player")
        ctx.user_message("Question", name="Player", canonical_name="Player")
        ctx.assistant_message(
            "",
            tool_calls=[
                {"id": "tc1", "function": {"name": "recall", "arguments": "{}"}}
            ],
            name="Bob",
            canonical_name="Bob",
        )
        ctx.tool_message("Bob's memory", "tc1")
        ctx.assistant_message("I see", tool_calls=[], name="Alice", canonical_name="Alice")

        branch = ctx.branch()
        view = branch.curated_view("Alice")
        roles = [m["role"] for m in view]
        assert roles == ["user", "assistant"]
        assert "Player says: Question" in view[0]["content"]
        assert "Bob attempts recall" in view[0]["content"]
        assert "Bob's memory" in view[0]["content"]
        assert view[1]["content"] == "I see"

    def test_curated_view_preserves_system_messages(self) -> None:
        """System messages survive reshaping untouched."""
        ctx = ConversationContext("Alice", "Player")
        ctx.enter_entities("Alice", "Player")
        ctx.context.append({"role": "system", "content": "Directive"})
        ctx.user_message("Hi", name="Player", canonical_name="Player")
        ctx.assistant_message("Hello", tool_calls=[], name="Alice", canonical_name="Alice")

        branch = ctx.branch()
        view = branch.curated_view("Alice")
        roles = [m["role"] for m in view]
        assert roles == ["system", "user", "assistant"]
        assert view[0]["content"] == "Directive"

    def test_curated_view_for_orchestrator(self) -> None:
        """Messages tagged as __orchestrator__ are treated as the observer."""
        ctx = ConversationContext("Alice", "Player")
        ctx.enter_entities("Alice", "Player")
        ctx.user_message("Hi", name="Player", canonical_name="Player")
        ctx.assistant_message(
            "",
            tool_calls=[
                {"id": "tc1", "function": {"name": "next_round", "arguments": "{}"}}
            ],
            name="System",
            canonical_name="__orchestrator__",
        )
        ctx.tool_message("done", "tc1")
        ctx.assistant_message("Hello", tool_calls=[], name="Alice", canonical_name="Alice")

        branch = ctx.branch()
        view = branch.curated_view("__orchestrator__")
        roles = [m["role"] for m in view]
        assert roles == ["user", "assistant", "tool", "user"]
        assert view[0]["content"] == "Player says: Hi"
        assert view[1].get("tool_calls")
        assert view[3]["content"] == "Alice says: Hello"

    def test_curated_view_orchestrator_separate_messages(self) -> None:
        """With collapse=False, each non-observer message becomes its own user message."""
        ctx = ConversationContext("Alice", "Bob", "Player")
        ctx.enter_entities("Alice", "Bob", "Player")
        ctx.user_message("Hi", name="Player", canonical_name="Player")
        ctx.assistant_message("Hello", tool_calls=[], name="Alice", canonical_name="Alice")
        ctx.user_message("Hey", name="Bob", canonical_name="Bob")
        ctx.assistant_message(
            "",
            tool_calls=[
                {"id": "tc1", "function": {"name": "next_round", "arguments": "{}"}}
            ],
            name="System",
            canonical_name="__orchestrator__",
        )

        branch = ctx.branch()
        view = branch.curated_view("__orchestrator__", collapse=False)
        user_contents = [m["content"] for m in view if m["role"] == "user"]
        assert "Player says: Hi" in user_contents
        assert "Alice says: Hello" in user_contents
        assert "Bob says: Hey" in user_contents
        # The observer's own tool-call turn stays assistant.
        assert any(m["role"] == "assistant" and m.get("tool_calls") for m in view)


class TestToNarrativeText:
    """Tests for the thin transcript formatter used by summarizers/finalizers."""

    def test_formats_user_assistant_and_system(self) -> None:
        """User lines keep their speaker prefix; observer assistant lines are labeled."""
        messages = [
            {"role": "system", "content": "Status update."},
            {"role": "user", "content": "Player says: Hi"},
            {"role": "assistant", "content": "Hello", "name": "Orchestrator"},
        ]
        text = ConversationContext.to_narrative_text(messages, observer_name="Orchestrator")
        assert "System: Status update." in text
        assert "Player says: Hi" in text
        assert "Orchestrator: Hello" in text

    def test_pairs_observer_tool_calls_with_results(self) -> None:
        """Observer tool-call turns are rendered as readable used/attempted lines."""
        messages = [
            {"role": "user", "content": "Player says: Go"},
            {
                "role": "assistant",
                "content": "",
                "name": "Orchestrator",
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "next_round", "arguments": '{"count": 1}'}}
                ],
            },
            {"role": "tool", "content": "Done", "tool_call_id": "tc1"},
        ]
        text = ConversationContext.to_narrative_text(messages, observer_name="Orchestrator")
        assert "Orchestrator used next_round" in text
        assert "Done" in text

    def test_strips_leaked_dsml_markup(self) -> None:
        """Raw DSML tool-call blocks are removed from content."""
        messages = [
            {
                "role": "assistant",
                "content": "Hmm <｜｜DSML｜｜tool_calls>recall</｜｜DSML｜｜> maybe not.",
                "name": "Orchestrator",
            },
        ]
        text = ConversationContext.to_narrative_text(messages, observer_name="Orchestrator")
        assert "DSML" not in text
        assert "Orchestrator: Hmm  maybe not." in text

    def test_respects_max_lines(self) -> None:
        """Only the last N narrative lines are kept."""
        messages = [
            {"role": "user", "content": "Player says: one"},
            {"role": "user", "content": "Player says: two"},
            {"role": "user", "content": "Player says: three"},
        ]
        text = ConversationContext.to_narrative_text(messages, max_lines=2)
        assert "one" not in text
        assert "two" in text
        assert "three" in text
