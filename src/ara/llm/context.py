"""Conversation context manager with slice-based entity visibility.

Each entity (character) only sees messages from rounds in which they were
present in the scene.  This mimics real-world conversations where off-scene
characters miss dialogue.
"""

from __future__ import annotations

import json
import operator
import re
from functools import reduce
from typing import Self

from openai.types.chat import ChatCompletionMessageParam

from ara.llm.models import Context
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
        """Create a conversation context.

        :param entities: Names of all entities that may participate across the
            lifetime of this context.
        :param injected_context: Messages prepended to every view.
        :param tmp_from: If given, create a temporary *branch* copying state
            from another instance.  Branches can be mutated without affecting
            the base.
        """
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
        logger.debug(f"ConversationContext entities={list(self.seen_entities)} present={list(self.present_entities)}")

    def branch(self) -> Self:
        """Return a temporary copy of this context.

        Changes to the branch do not affect the original.
        """
        return self.__class__(tmp_from=self)

    @staticmethod
    def is_visible(msg: dict, observer: str) -> bool:
        """Whether *observer* may perceive *msg* under hidden/visible_to rules.

        A message marked ``_hidden`` is visible only to its sender and to
        observers listed in its ``_visible_to`` set; every other message is
        visible to everyone.  This is the single visibility rule used by all
        readers (``context_of`` and the engine's branch filters).
        """
        if not msg.get("_hidden"):
            return True
        visible_to = set(msg.get("_visible_to") or [])
        sender = msg.get("_canonical_name") or msg.get("name")
        return observer == sender or observer in visible_to

    def context_of(self, entity: str) -> Context:
        """Return the sub-sequence of messages visible to *entity*.

        Messages sent by a hidden character are excluded unless *entity* is the
        sender or is listed in the message's ``visible_to`` set.

        After filtering, the view is normalized so that it remains a legal chat
        sequence: adjacent ``user``/``assistant`` turns are separated by empty
        padding messages, orphaned ``tool`` results are dropped, and
        ``assistant`` messages whose tool results were lost have their
        ``tool_calls`` stripped.

        :param entity: Entity name.
        :return: Filtered, normalized message list.
        :raises RuntimeError: If *entity* is not tracked.
        """
        try:
            slices = self.seen_entities[entity]
            messages = reduce(operator.add, (self.context[sl] for sl in slices), [])
        except KeyError as e:
            raise RuntimeError(f"{entity} not in {set(self.seen_entities)}") from e

        cleaned: Context = []
        for msg in messages:
            if not self.is_visible(msg, entity):
                continue
            # Strip private visibility markers before returning.
            cleaned.append(
                {k: v for k, v in msg.items() if not k.startswith("_")}
            )
        logger.debug(f"Context for {entity}: {len(cleaned)} raw messages; present={list(self.present_entities)}")
        return self._normalize_view(cleaned)

    def curated_view(self, observer: str, collapse: bool = True) -> Context:
        """Return a single-assistant view where *observer* is the only assistant.

        All messages sent by *observer* keep their original role (``assistant``)
        and structure, including tool calls and tool results. Every other speaker
        is reported using canonical labels:

        - ``"{Name} says: {content}"`` for normal speech.
        - ``"{Name} attempts {tool_name}"`` for tool calls.
        - ``"  -> {result}"`` for the matching tool result.

        When *collapse* is ``True`` (default for characters), adjacent messages
        from other speakers are folded into a single ``user`` block. When
        *collapse* is ``False`` (recommended for the orchestrator), each source
        message becomes its own ``user`` message so per-turn KV-cache boundaries
        are preserved.

        ``system`` messages are preserved. The result is normalized so it remains
        a valid chat-completion sequence.

        :param observer: The display or canonical name of the observing entity.
        :param collapse: Whether to merge consecutive non-observer messages.
        :return: Reshaped, normalized message list.
        """
        out: Context = []
        user_lines: list[str] = []
        observer_pending_ids: set[str] = set()
        other_attempt_indices: dict[str, int] = {}

        def _flush_user_block() -> None:
            if user_lines:
                out.append({"role": "user", "content": "\n".join(user_lines)})
                user_lines.clear()
                other_attempt_indices.clear()

        def _speaker(msg: dict) -> str:
            return msg.get("_canonical_name") or msg.get("name") or "System"

        def _add_user_line(line: str, tool_call_id: str | None = None) -> int:
            """Add a canonical user line and return its index for result pairing."""
            if collapse:
                user_lines.append(line)
                return len(user_lines) - 1
            out.append({"role": "user", "content": line})
            return len(out) - 1

        for msg in self.context:
            role = msg.get("role")
            if role == "system":
                if collapse:
                    _flush_user_block()
                out.append({k: v for k, v in msg.items() if not k.startswith("_")})
                continue

            speaker = _speaker(msg)
            is_observer = speaker == observer

            if role == "assistant":
                if is_observer:
                    if collapse:
                        _flush_user_block()
                    out.append({k: v for k, v in msg.items() if not k.startswith("_")})
                    tool_calls = msg.get("tool_calls") or []
                    observer_pending_ids = {
                        tc.get("id") for tc in tool_calls if tc.get("id")
                    }
                    other_attempt_indices.clear()
                else:
                    content = msg.get("content") or ""
                    if content:
                        _add_user_line(f"{speaker} says: {content}")
                    for tc in msg.get("tool_calls") or []:
                        if not isinstance(tc, dict):
                            continue
                        name = tc.get("function", {}).get("name", "tool")
                        idx = _add_user_line(f"{speaker} attempts {name}")
                        tc_id = tc.get("id")
                        if tc_id:
                            other_attempt_indices[tc_id] = idx
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id in observer_pending_ids:
                    if collapse:
                        _flush_user_block()
                    out.append({k: v for k, v in msg.items() if not k.startswith("_")})
                    observer_pending_ids.discard(tool_call_id)
                    if not observer_pending_ids:
                        observer_pending_ids = set()
                elif tool_call_id in other_attempt_indices:
                    idx = other_attempt_indices.pop(tool_call_id)
                    result = (msg.get("content") or "").strip()
                    if result:
                        if collapse:
                            user_lines[idx] += f"\n  -> {result}"
                        else:
                            out[idx]["content"] += f"\n  -> {result}"
                else:
                    logger.debug(
                        "Dropping orphaned tool result in curated_view (tool_call_id=%s)",
                        tool_call_id,
                    )
                continue

            if role == "user":
                content = msg.get("content") or ""
                if content:
                    _add_user_line(f"{speaker} says: {content}")
                continue

            # Unknown role: preserve separately.
            if collapse:
                _flush_user_block()
            out.append({k: v for k, v in msg.items() if not k.startswith("_")})

        if collapse:
            _flush_user_block()
        return self._normalize_curated_view(out)

    @staticmethod
    def _normalize_curated_view(messages: Context) -> Context:
        """Normalize a curated view without breaking tool-call sequences.

        Unlike :meth:`_normalize_view`, this allows ``assistant -> tool ->
        assistant`` sequences (the observer's own multi-step tool turn) and only
        inserts padding for truly adjacent same-role turns.
        """
        out: Context = []
        last_role: str | None = None
        last_was_tool = False

        def _padding(role: str) -> dict[str, Any]:
            msg: dict[str, Any] = {"role": role, "content": ""}
            if role == "user":
                msg["name"] = ConversationContext.default_sysname
            return msg

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                out.append(msg)
                last_role = "system"
                last_was_tool = False
                continue

            if role == "tool":
                out.append(msg)
                last_role = "assistant"
                last_was_tool = True
                continue

            if role == "assistant":
                if last_role == "assistant" and not last_was_tool:
                    out.append(_padding("user"))
                out.append(msg)
                last_role = "assistant"
                last_was_tool = False
                continue

            if role == "user":
                if last_role == "user":
                    out.append(_padding("assistant"))
                out.append(msg)
                last_role = "user"
                last_was_tool = False
                continue

            out.append(msg)
            last_role = role
            last_was_tool = False

        return out

    @staticmethod
    def _normalize_view(messages: Context) -> Context:
        """Ensure a filtered message view is a valid chat-completion sequence.

        Removing hidden messages can create adjacent turns of the same role or
        leave ``tool`` results without the matching ``assistant`` tool call.
        This method inserts empty padding turns where required and cleans up
        dangling tool-call references.
        """
        out: Context = []
        last_role: str | None = None
        pending_assistant_idx: int | None = None
        pending_tool_ids: set[str] = set()

        def _strip_pending_tool_calls() -> None:
            nonlocal pending_assistant_idx, pending_tool_ids
            if pending_assistant_idx is None or not pending_tool_ids:
                return
            assistant_msg = dict(out[pending_assistant_idx])
            assistant_msg.pop("tool_calls", None)
            out[pending_assistant_idx] = assistant_msg
            pending_assistant_idx = None
            pending_tool_ids = set()

        def _padding(role: str) -> dict[str, Any]:
            msg: dict[str, Any] = {"role": role, "content": ""}
            if role == "user":
                msg["name"] = ConversationContext.default_sysname
            return msg

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                out.append(msg)
                continue

            if role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if (
                    pending_assistant_idx is not None
                    and tool_call_id in pending_tool_ids
                ):
                    out.append(msg)
                    pending_tool_ids.discard(tool_call_id)
                    if not pending_tool_ids:
                        pending_assistant_idx = None
                    last_role = "assistant"
                else:
                    logger.debug(
                        "Dropping orphaned tool result (tool_call_id=%s)",
                        tool_call_id,
                    )
                continue

            if role == "assistant":
                _strip_pending_tool_calls()
                if last_role == "assistant":
                    out.append(_padding("user"))
                tool_calls = msg.get("tool_calls") or []
                tool_ids = {tc.get("id") for tc in tool_calls if tc.get("id")}
                out.append(msg)
                last_role = "assistant"
                if tool_ids:
                    pending_assistant_idx = len(out) - 1
                    pending_tool_ids = tool_ids
                continue

            if role == "user":
                if last_role == "user":
                    out.append(_padding("assistant"))
                out.append(msg)
                last_role = "user"
                continue

            # Unknown role: preserve but do not use for alternation logic.
            out.append(msg)

        _strip_pending_tool_calls()
        return out

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

    def add_entities(self, *entities: str) -> None:
        """Add new entities to the conversation mid-scene.

        Newly added entities can only see messages from this point forward.
        This is used for anonymous characters spawned on the fly.

        :param entities: Names of new entities to track.
        """
        for entity in entities:
            if entity in self.seen_entities:
                continue
            self.seen_entities[entity] = [slice(len(self.context), None)]
        if entities:
            logger.debug(f"Added entities to conversation: {entities}")
            self.present_entities |= set(entities)

    def user_message(
        self,
        content: str,
        name: str = default_sysname,
        hidden: bool = False,
        visible_to: set[str] | None = None,
        canonical_name: str | None = None,
    ) -> Self:
        """Append a user message.

        If the previous message was not from an assistant or tool, an empty
        assistant padding message is inserted automatically to preserve valid
        API sequencing.

        :param content: Message text.
        :param name: Speaker display name.
        :param hidden: If True, this message is only visible to the speaker and
            the names in *visible_to*.
        :param visible_to: Canonical names of observers that can perceive a hidden message.
        :param canonical_name: Stable internal identifier for the speaker.
        :return: Self for chaining.
        """
        if self.head is not None and self.head.get("role") not in ("assistant", "tool"):
            self.assistant_message("", tool_calls=[], name=self.default_sysname)
        self.head = {
            "role": "user",
            "content": content,
            "name": name,
            "_canonical_name": canonical_name or name,
        }
        if hidden:
            self.head["_hidden"] = True
            self.head["_visible_to"] = list(visible_to or [])
        self.context.append(self.head)
        return self

    def tool_message(self, content: str, tool_call_id: str) -> Self:
        """Append a tool-result message.

        :param content: Result text.
        :param tool_call_id: ID of the tool call being answered.
        :return: Self for chaining.
        :raises RuntimeError: If the previous message was not an assistant tool
            call or another tool result.
        """
        if self.head is None or self.head.get("role") not in ("assistant", "tool"):
            raise RuntimeError("Tool message must follow an assistant tool call or another tool result.")
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
        hidden: bool = False,
        visible_to: set[str] | None = None,
        canonical_name: str | None = None,
    ) -> Self:
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
        :param name: Speaker display name.
        :param reasoning_content: Raw chain-of-thought text (may be empty).
        :param hidden: If True, this message is only visible to the speaker and
            the names in *visible_to*.
        :param visible_to: Canonical names of observers that can perceive a hidden message.
        :param canonical_name: Stable internal identifier for the speaker.
        :return: Self for chaining.
        """
        # DeepSeek thinking mode may produce content=None when only
        # reasoning_content is present.
        if content is None:
            content = ""

        if self.head is not None and self.head.get("role") == "assistant":
            self.user_message("The scene continues.", name=self.default_sysname)

        msg: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": content,
            "name": name,
            "_canonical_name": canonical_name or name,
        }
        if hidden:
            msg["_hidden"] = True
            msg["_visible_to"] = list(visible_to or [])

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

    def speech_message(
        self,
        content: str,
        speaker: Any,
        *,
        role: str = "assistant",
        tool_calls: list[dict] | None = None,
        reasoning_content: str = "",
    ) -> Self:
        """Append a character speech message with flags derived from *speaker*.

        *speaker* is any object with ``name``, ``canonical_name``, ``hidden``
        and ``visible_to`` attributes (e.g. a character).  Every speech write
        site should use this helper so hidden/visible_to flags stay consistent
        (an unflagged message from a hidden character leaks to all readers).

        :param content: Message text.
        :param speaker: The speaking character.
        :param role: ``"assistant"`` for NPC/narrator speech, ``"user"`` for
            player input.
        :param tool_calls: Completed tool-call descriptors (assistant only).
        :param reasoning_content: Raw chain-of-thought text (assistant only).
        :return: Self for chaining.
        """
        kwargs: dict[str, Any] = {
            "name": speaker.name,
            "hidden": speaker.hidden,
            "visible_to": set(speaker.visible_to) if speaker.hidden else None,
            "canonical_name": speaker.canonical_name,
        }
        if role == "user":
            return self.user_message(content, **kwargs)
        return self.assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            **kwargs,
        )

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
            hidden = bool(line.get("_hidden"))
            visible_to = set(line.get("_visible_to") or [])
            if role == "system":
                self.context.append(dict(line))
                continue
            if role == "user":
                self.user_message(content, name=name, hidden=hidden, visible_to=visible_to)
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
                    hidden=hidden,
                    visible_to=visible_to,
                )
            elif role == "tool":
                self.tool_message(
                    content,
                    tool_call_id=line["tool_call_id"],  # type: ignore[typeddict-item]
                    hidden=bool(line.get("_hidden")),
                    visible_to=set(line.get("_visible_to") or []),
                )
        return self

    def to_list(self, entity: str | None = None) -> Context:
        """Return the full message list for an API call.

        :param entity: If given, only messages visible to that entity are
            included after the injected context.
        :return: Combined message list with private visibility markers removed.
        """
        messages = self.context_of(entity) if entity else self.context
        cleaned = [{k: v for k, v in msg.items() if not k.startswith("_")} for msg in messages]
        return self.injected_context + cleaned

    @staticmethod
    def to_narrative_text(
        messages: Context,
        observer_name: str = "Orchestrator",
        max_lines: int | None = None,
    ) -> str:
        """Convert a reshaped message list into a plain speaker-labeled transcript.

        This is a thin formatter, not a reshaper. It is intended to be used on
        the output of :meth:`curated_view`, where one observer is the only
        ``assistant`` and every other speaker is represented through ``user``
        messages.

        :param messages: Already-curated message list.
        :param observer_name: Display name to use for the observer's own turns.
        :param max_lines: If given, keep only the last N narrative lines.
        :return: Plain-text transcript suitable for prompts or sub-agents.
        """

        def _strip_dsml(text: str) -> str:
            """Remove any leaked DSML tool-call markup."""
            text = re.sub(
                r"<\uff5c\uff5cDSML\uff5c\uff5c[^>]*>.*?</\uff5c\uff5cDSML\uff5c\uff5c>",
                "",
                text,
                flags=re.DOTALL,
            )
            text = re.sub(r"<\uff5c\uff5cDSML\uff5c\uff5c[^>]*>", "", text)
            return text.strip()

        def _pretty_args(args: str) -> str:
            if not args:
                return ""
            try:
                obj = json.loads(args)
                compact = json.dumps(obj, ensure_ascii=False)
            except Exception:
                compact = args
            if len(compact) > 120:
                compact = compact[:117] + "..."
            return compact

        # First pass: collect tool results by tool_call_id.
        tool_results: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id:
                    tool_results[tc_id] = (msg.get("content") or "").strip()

        lines: list[str] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                content = (msg.get("content") or "").strip()
                if content:
                    lines.append(f"System: {content}")
                continue

            if role == "user":
                content = (msg.get("content") or "").strip()
                if content:
                    lines.append(content)
                continue

            if role == "assistant":
                content = _strip_dsml(msg.get("content") or "").strip()
                if content:
                    lines.append(f"{observer_name}: {content}")
                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name", "tool")
                    args = _pretty_args(fn.get("arguments", ""))
                    tc_id = tc.get("id")
                    result = tool_results.get(tc_id, "")
                    if result:
                        if len(result) > 200:
                            result = result[:197] + "..."
                        lines.append(f"{observer_name} used {name}({args}): {result}")
                    else:
                        lines.append(f"{observer_name} attempted {name}({args})")
                continue

            # tool messages are consumed via tool_results and skipped here.

        if max_lines is not None:
            lines = lines[-max_lines:]
        return "\n".join(lines)
