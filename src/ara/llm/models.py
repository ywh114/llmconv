"""Shared types and enums for the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class GameRole(StrEnum):
    """Enumeration of LLM roles used within the engine.

    Each role receives distinct temperature and system-prompt settings.
    """

    CHARACTER = auto()
    """An in-world NPC or the player character."""

    NARRATOR = auto()
    """The visual-novel narrator who describes environment and mood."""

    ORCHESTRATOR = auto()
    """The director/DM that decides who speaks next and how the scene advances."""

    SUMMARIZER = auto()
    """Scene-transition summarizer that bridges context between scenes."""


@dataclass
class StreamResult:
    """Accumulator for a streamed LLM response.

    :ivar content: Final response text produced by the model.
    :ivar reasoning_content: Chain-of-thought text when DeepSeek thinking mode
        is enabled.  May be empty for non-thinking models.
    :ivar tool_calls: Parsed tool-call descriptors accumulated from delta chunks.
    """

    content: str = ""
    """Final response text."""

    reasoning_content: str = ""
    """Chain-of-thought content from the model."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """Tool calls extracted from the streamed response."""

    usage: dict[str, Any] = field(default_factory=dict)
    """Provider usage metadata, e.g. prompt/completion/cache-hit tokens."""

    def has_tool_calls(self) -> bool:
        """Return ``True`` if the response contains at least one tool call."""
        return bool(self.tool_calls)


Context = list["ChatCompletionMessageParam"]
"""Type alias for a list of chat-completion message parameters."""
