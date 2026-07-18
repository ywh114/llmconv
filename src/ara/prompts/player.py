"""Player-side prompt builders."""

from __future__ import annotations

from ara.world.character import Character
from ara.world.scene import Scene


def player_input_prompt(player: Character, scene: Scene, suggestion: str) -> str:
    """Build the prompt that expands a suggestion into spoken player dialogue.

    :param player: The player character.
    :param scene: Current scene (provides zeitgeist, tone, language).
    :param suggestion: The intent to phrase (e.g. "Ask about the book").
    :return: System prompt for the one-off generation call.
    """
    return (
        f"You are {player.name} in a visual novel scene.\n"
        f"Zeitgeist: {scene.zeitgeist}\n"
        f"Tone: {scene.tone}\n"
        f"Language: {scene.language}\n\n"
        f"Write a brief, natural spoken response (1–3 sentences) "
        f"that matches this intent: {suggestion}\n\n"
        f"Respond with ONLY the dialogue. No quotes, no narration, "
        f"no stage directions."
    )
