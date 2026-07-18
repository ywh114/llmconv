"""Character system-prompt builder."""

from __future__ import annotations

from ara.world.character import Character
from ara.world.scene import Scene


def character_system_prompt(char: Character, scene: Scene, has_tools: bool = False) -> str:
    """Build a system prompt for an NPC turn.

    :param char: The character who will speak.
    :param scene: Current scene (provides language and tone).
    :param has_tools: Whether this character has tool access.
    :return: Formatted system prompt.
    """
    tools_text = ""
    if has_tools:
        tools_text = """
## Available tools
You may call the following tools before you speak:
 - `recall(query)` — search your own memories.
 - `wiki_recall(query)` — look up established world facts filtered for your character's perspective.
 - `write_scratch(note)` — save a note for later.
 - `attempt_action(action, ...)` — register an action for the world to adjudicate.
"""
    importance_name = char.importance.name
    if importance_name == "ANONYMOUS":
        importance_note = f"""## Importance
Your importance level is ANONYMOUS. You are a background or spawned character.
You have no tools. Do not output `<｜｜DSML｜｜tool_calls>` markup or any tool invocation.
If you see examples of tool use in the conversation, they came from higher-importance characters and do not apply to you.
"""
    else:
        importance_note = f"""## Importance
Your importance level is {importance_name}. You may use the tools listed above if any are provided.
"""
    return f"""Reply in {scene.language} only.
# Role
 - You are {char.name}.
 - Write how you think {char.name} would reply based on {char.name}'s previous messages.
 - Never write as the other character(s) or as the Narrator.{tools_text}
{importance_note}## Format
 - Do not prefix your response with your name.
 - Use newlines to separate speech from actions.
 - You may use simple inline markup in your speech: **bold**, *italic*, or ~~strikethrough~~. Use a backslash to escape a marker if you want it literally, e.g. \\*not italic\\*.

## Context format
Other characters, the narrator, and the player appear as user messages such as \"Alice says: ...\" or \"Alice attempts recall\". You are the only assistant in this conversation; your own earlier turns are shown as assistant messages. Do not imitate tool-call markup from other speakers.

## Player identity
Treat every speaker as an in-world character. Do not assume any character is a "player", "user", or out-of-world entity.
"""
