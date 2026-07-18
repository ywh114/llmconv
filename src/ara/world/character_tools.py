"""Per-character tool schemas and handlers for NPC turns.

Characters at :attr:`Importance.IMPORTANT` or higher get four tools during
their turns: ``recall`` (personal memory), ``wiki_recall`` (world facts),
``write_scratch`` (private notes), and ``attempt_action`` (deferred action
adjudication by the orchestrator).  The schemas are constant, so they are
built once at module level; only the handlers capture per-character state.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ara.llm.tools import ToolRegistry, tool
from ara.world.character import Character, Importance

_RECALL_TOOL = tool(
    name="recall",
    description=(
        "Search YOUR OWN memory for relevant past conversations or events. "
        "This only returns memories from your personal perspective - you cannot "
        "recall things you did not personally experience or store."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "What you want to remember. E.g., 'What did Player say about the book?'",
        }
    },
    required=["query"],
    strict=True,
)

_WIKI_RECALL_TOOL = tool(
    name="wiki_recall",
    description=(
        "Look up established world facts from the permanent wiki. Use this when you need "
        "to know something about the world, setting, factions, history, or rules that your "
        "character could reasonably know or have heard of. The result is filtered for your "
        "character's perspective and expertise."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "What you want to know about the world. E.g., 'What are the major sects in this city?'",
        }
    },
    required=["query"],
    strict=True,
)

_WRITE_SCRATCH_TOOL = tool(
    name="write_scratch",
    description="Write a note to your scratchpad for future reference.",
    properties={
        "note": {
            "type": "string",
            "description": "The note to save. This will be visible to you in future scenes.",
        }
    },
    required=["note"],
    strict=True,
)

_ATTEMPT_ACTION_TOOL = tool(
    name="attempt_action",
    description=(
        "Record an action you want to attempt. The orchestrator will see this "
        "on the next turn and decide the outcome. Use for uncertain actions, "
        "stealth, combat, or anything the world model should adjudicate."
    ),
    properties={
        "action": {
            "type": "string",
            "description": "What you are trying to do.",
        },
        "intent": {
            "type": "string",
            "description": "Why you are doing it or what outcome you want.",
        },
        "target": {
            "type": "string",
            "description": "Who or what the action is directed at, if any.",
        },
        "secrecy": {
            "type": "string",
            "enum": ["silent", "quiet", "loud", "obvious"],
            "description": "How noticeable the action is.",
        },
    },
    required=["action"],
    strict=True,
)

CHARACTER_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _RECALL_TOOL,
    _WIKI_RECALL_TOOL,
    _WRITE_SCRATCH_TOOL,
    _ATTEMPT_ACTION_TOOL,
]


def register_character_tools(
    char: Character,
    registry: ToolRegistry,
    *,
    recall_fn: Callable[[str], list[str]],
    wiki_fn: Callable[[str], str],
    record_attempt_fn: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """Register *char*'s tools on *registry* and return their schemas.

    Characters below :attr:`Importance.IMPORTANT` get no tools (empty list).

    :param recall_fn: Query -> personal memory lines for *char*.
    :param wiki_fn: Query -> wiki facts filtered for *char*'s perspective.
    :param record_attempt_fn: Append an action-attempt dict (a ``source`` key
        with *char*'s canonical name is added before recording).
    """
    if char.importance < Importance.IMPORTANT:
        return []

    def _recall_handler(args: str) -> str:
        data = json.loads(args)
        memories = recall_fn(data.get("query", ""))
        if memories:
            return "\n".join(f"- {m}" for m in memories)
        return "You don't recall anything relevant."

    def _wiki_recall_handler(args: str) -> str:
        data = json.loads(args)
        return wiki_fn(data.get("query", ""))

    def _write_scratch_handler(args: str) -> str:
        data = json.loads(args)
        note = data.get("note", "")
        if note:
            if char.scratch.is_empty():
                char.scratch.text = f"[Note]: {note}"
            else:
                char.scratch.text += f"\n[Note]: {note}"
        return "Note saved."

    def _attempt_action_handler(args: str) -> str:
        data = json.loads(args)
        data["source"] = char.canonical_name
        record_attempt_fn(data)
        return "Action attempt recorded for the orchestrator."

    registry.register("recall", _recall_handler)
    registry.register("wiki_recall", _wiki_recall_handler)
    registry.register("write_scratch", _write_scratch_handler)
    registry.register("attempt_action", _attempt_action_handler)
    return list(CHARACTER_TOOL_SCHEMAS)
