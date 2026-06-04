"""Conversation context manager with slice-based entity visibility.

Each entity (character) only sees messages from rounds in which they were
present in the scene.  This mimics real-world conversations where off-scene
characters miss dialogue.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import Self

from openai.types.chat import ChatCompletionMessageParam

from ara.models import Context
from ara.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationContext:
    """Manages multi-round conversation history with per-entity visibility.

    :param entities: Names of all entities that may participate across the
        lifetime of this context.
    :param injected_context: Messages prepended to every view (e.g. system
        prompts, plot summaries).
    :param tmp_from: If given, create a temporary *branch* copying state from
        another instance.  Branches can be mutated without affecting the base.
    """

    default_sysname = "System"
    """Default name used for system-oriented padding messages."""

    def __init__(
        self,
        *entities: str,
        injected_context: Context | None = None,
        tmp_from: Self | None = None,
    ) -> None:
        self.base = tmp_from is None
        if self.base:
            self.injected_context: Context = injected_context or []
            self.context: Context = []
            self.head: ChatCompletionMessageParam | None = None
            self.seen_entities: dict[str, list[slice]] = {
                entity: [] for entity in entities
            }
            self.present_entities: set[str] = set()
        else:
            assert tmp_from is not None
            self.injected_context = list(tmp_from.injected_context)
            self.context = list(tmp_from.context)
            self.head = tmp_from.head.copy() if tmp_from.head else None
            self.present_entities = set(tmp_from.present_entities)
            self.seen_entities = {
                k: list(v) for k, v in tmp_from.seen_entities.items()
            }

    def branch(self) -> Self:
        """Return a temporary copy of this context.

        Changes to the branch do not affect the original.
        """
        return self.__class__(tmp_from=self)

    def context_of(self, entity: str) -> Context:
        """Return the sub-sequence of messages visible to *entity*.

        :param entity: Entity name.
        :return: Filtered message list.
        :raises RuntimeError: If *entity* is not tracked.
        """
        try:
            slices = self.seen_entities[entity]
            return reduce(operator.add, (self.context[sl] for sl in slices), [])
        except KeyError as e:
            raise RuntimeError(f"{entity} not in {set(self.seen_entities)}") from e

    def filter_to(self, entity: str) -> None:
        """Destructively filter the working context to *entity*'s view.

        This is only allowed on a :meth:`branch`, not on the base context.

        :raises RuntimeError: If called on a base context.
        """
        if self.base:
            raise RuntimeError(
                "Cannot filter base context; use branch() first."
            )
        self.context = self.context_of(entity)
        self.head = self.context[-1] if self.context else None

    def enter_entities(self, *entities: str) -> None:
        """Mark *entities* as entering the scene.

        From this message index onward, the entities will see new messages.

        :raises RuntimeError: If an entity is not tracked.
        """
        for entity in entities:
            if entity not in self.seen_entities:
                raise RuntimeError(f"{entity} is not in the scene.")
            slices = self.seen_entities[entity]
            if slices and slices[-1].stop is None:
                logger.warning(f"{entity} is already present, ignoring.")
                continue
            self.seen_entities[entity] = slices + [slice(len(self.context), None)]
        if entities:
            logger.debug(f"Entered conversation: {entities}")
        self.present_entities |= set(entities)

    def exit_entities(self, *entities: str) -> None:
        """Mark *entities* as exiting the scene.

        Their current visibility slice is closed at the current message index.

        :raises RuntimeError: If an entity is not tracked.
        """
        for entity in entities:
            if entity not in self.seen_entities:
                raise RuntimeError(f"{entity} is not in the scene.")
            slices = self.seen_entities[entity]
            if not slices or slices[-1].stop is not None:
                logger.warning(f"{entity} is already off-scene, ignoring.")
                continue
            self.seen_entities[entity][-1] = slice(
                slices[-1].start, len(self.context)
            )
        if entities:
            logger.debug(f"Exited conversation: {entities}")
        self.present_entities -= set(entities)

    def user_message(self, content: str, name: str = default_sysname) -> Self:
        """Append a user message.

        If the previous message was not from an assistant or tool, an empty
        assistant padding message is inserted automatically to preserve valid
        API sequencing.

        :param content: Message text.
        :param name: Speaker name.
        :return: Self for chaining.
        """
        if self.head is not None and self.head.get("role") not in ("assistant", "tool"):
            self.assistant_message("", tool_calls=[], name=self.default_sysname)
        self.head = {
            "role": "user",
            "content": content,
            "name": name,
        }
        self.context.append(self.head)
        return self

    def tool_message(self, content: str, tool_call_id: str) -> Self:
        """Append a tool-result message.

        :param content: Result text.
        :param tool_call_id: ID of the tool call being answered.
        :return: Self for chaining.
        :raises RuntimeError: If the previous message was not an assistant tool
            call.
        """
        if self.head is None or self.head.get("role") != "assistant":
            raise RuntimeError("Tool message must follow assistant tool call.")
        self.head = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }
        self.context.append(self.head)
        return self

    def assistant_message(
        self,
        content: str,
        tool_calls: list[dict] | None = None,
        name: str = default_sysname,
        reasoning_content: str = "",
    ) -> Self:
        # DeepSeek thinking mode may produce content=None when only
        # reasoning_content is present.
        if content is None:
            content = ""
        """Append an assistant message.

        DeepSeek thinking-mode compliance is handled automatically:

        * If *tool_calls* is non-empty, ``reasoning_content`` is **preserved**
          in the message dict because the API requires it on subsequent turns
          that involve tool calls.
        * If *tool_calls* is empty, ``reasoning_content`` is **omitted**
          because the API ignores it on non-tool turns.

        If the previous message was from an assistant, an empty user padding
        message is inserted automatically to preserve valid API sequencing.

        :param content: Assistant text.
        :param tool_calls: List of completed tool-call descriptors.
        :param name: Speaker name.
        :param reasoning_content: Raw chain-of-thought text (may be empty).
        :return: Self for chaining.
        """
        if self.head is not None and self.head.get("role") == "assistant":
            self.user_message("The scene continues.", name=self.default_sysname)

        msg: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": content,
            "name": name,
        }

        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls
            ]
        # DeepSeek rule: reasoning_content must be preserved for tool-call turns.
        # Including it on non-tool turns is harmless (the API ignores it).
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content  # type: ignore[typeddict-unknown-key]

        self.head = msg
        self.context.append(self.head)
        return self

    def concat_context(self, context: Context) -> Self:
        """Replay an existing message list into this context.

        Each message is routed through the appropriate ``*_message`` method
        so that sequencing rules and reasoning-content stripping are applied.

        :param context: Messages to replay.
        :return: Self for chaining.
        """
        for line in context:
            role = line.get("role")
            # DeepSeek thinking mode may return content=None when only
            # reasoning_content is present. Default to empty string.
            content = line.get("content", "") or ""
            name = line.get("name", self.default_sysname)
            if role == "user":
                self.user_message(content, name=name)
            elif role == "assistant":
                tcs = line.get("tool_calls", [])
                tc_list = [
                    {
                        "id": tc["id"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tcs
                ]
                self.assistant_message(
                    content,
                    tool_calls=tc_list or None,
                    name=name,
                    reasoning_content=line.get("reasoning_content", ""),  # type: ignore[arg-type]
                )
            elif role == "tool":
                self.tool_message(
                    content, tool_call_id=line["tool_call_id"]  # type: ignore[typeddict-item]
                )
        return self

    def to_list(self, entity: str | None = None) -> Context:
        """Return the full message list for an API call.

        :param entity: If given, only messages visible to that entity are
            included after the injected context.
        :return: Combined message list.
        """
        return self.injected_context + (
            self.context_of(entity) if entity else self.context
        )
