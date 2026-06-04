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
