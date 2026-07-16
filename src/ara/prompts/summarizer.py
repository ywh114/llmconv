"""Summarizer system- and user-prompt builders."""

from __future__ import annotations

from typing import Any

from ara.world.scene import Scene


def summarizer_system_prompt(language: str, has_changelog: bool, has_player_status: bool) -> str:
    """Build the summarizer system prompt.

    :param language: Target language for the summary.
    :param has_changelog: Whether a mechanical changelog is present.
    :param has_player_status: Whether player-status DSL is present.
    :return: Formatted system prompt.
    """
    changelog_note = (
        "\nA 'Mechanical state changes' section is included below. Use it to know "
        "exactly what changed in the scene (time, location edits, inventory, etc.). "
        "Do not drop items, skills, or bars unless the changelog or plot text says so."
        if has_changelog
        else ""
    )
    player_status_note = (
        "\nA 'Player status' section shows the player's final status in a "
        "sectioned DSL. Copy it unchanged unless the plot or a timeskip explicitly modifies "
        "it. If you edit it, emit a PLAYER_STATUS block containing the COMPLETE new DSL, "
        "not just the changed keys. Example:\n"
        '{\n  "title": "Commander Status",\n'
        '  "sections": [\n'
        '    {"type": "bars", "items": [{"label": "HP", "value": 85, "max": 100, "color": "red"}]},\n'
        '    {"type": "inventory", "items": ["Tea Cup"]},\n'
        '    {"type": "skills", "items": ["Leadership"]}\n'
        '  ]\n}'
        if has_player_status
        else ""
    )
    return f"""IMPORTANT: Write in {language} only!
You are the Summarizer - a background agent that prepares narrative continuity
between scenes in a visual novel.

## Task
Read the transcript of the scene that just ended, the private scratchpads of
the characters who were in that scene, and the plot of the upcoming scene.
Produce ONE summary for EACH character who will appear in the next scene, plus
a finalized description for the primary location.

### Rules for summaries
1. **Follow both scenes' considerations.** The "Current scene considerations" and
   "Upcoming scene considerations" contain scene-specific rules (e.g., anti-spoiler
   rules, vocabulary restrictions, pacing notes). Apply them when writing summaries
   and status updates. Do not leak source-canon backstory or future events that
   have not yet been established in play.
2. **Characters who were already present** (continuing characters) receive a
   very short bridging note - just enough to cover time jumps or emotional
   residue. 1–2 sentences max.
3. **Characters who are NEW to the next scene** receive a fuller recap of the
   key facts, emotional state, and unresolved threads they would reasonably
   know about. 2–4 sentences.
4. **Respect secrets.** Use the scratchpads to know what each character secretly
   knows or plans.  Do NOT reveal a secret in a summary meant for a character
   who should not know it.
5. **Optionally leave a note for the orchestrator.** If there are unresolved
   threads, suspicious patterns, or anything the orchestrator of the next scene
   should be aware of, emit an ORCHESTRATOR_NOTE block with a short hint.
6. Write each summary as neutral narration from that character's point of view,
   NOT as dialogue.  Do NOT use meta-language like "the player" or "the scene".
7. **Natural trivial progression:** for characters who were off-screen, assume
   small things progress naturally over time (wounds heal, fatigue fades, moods
   soften, meals finish) unless the plot or stored status flags say otherwise.
   Do NOT emit STATUS blocks for these tiny changes; just account for them when
   writing re-entry summaries.
8. **Use the exact names from the Next scene cast.** Do not translate IDs or
   invent new names. The Player character and Narrator are listed separately;
   do NOT add them as SUMMARY entries.
9. **Anonymous/transient NPCs** are marked `[anonymous]` in the cast. For each
   one, emit an `ANONYMOUS <Name>:` block with a short description/sprite so
   the next scene can instantiate them. You may also emit a `SUMMARY` for them
   if they continue from the previous scene. Do not introduce extra anonymous
   NPCs beyond those listed in the cast.

### Finalized location description
Rewrite the current location description so any major permanent changes are
incorporated smoothly.  ONLY keep changes that alter the location in a lasting
way (e.g. a house burned down, a bridge collapsed).  Ignore minor incidental
changes (e.g. a dog walked in, someone moved a chair).  If no major changes
occurred, return the original description cleaned of any ``[Update]`` tags.
{changelog_note}
{player_status_note}

### Optional expand tool
If you need to know about characters not listed in the roster below, output ONLY:
QUERY: <search phrase>
You will receive a short list of matching characters and can then write the final
summary. Do not use this unless necessary.

Return your response in this exact format:

SUMMARY <CharacterName>:
<summary for this character>

(repeat for every character in the next scene)

Optionally, if a character's persona should be adjusted for the upcoming scene
(e.g., to respect an anti-spoiler rule or to reflect a major shift in mindset),
override one or more card fields:

CHARACTER <CharacterName>:
{{"summary": "...", "personality": "...", "scenario": "..."}}

Only include fields you want to override. Empty strings are ignored.

ANONYMOUS <CharacterName>:
{{"description": "...", "sprite": "..."}}

LOCATION:
<finalized location description>

TIME:
<new world time, or the original time if unchanged. One word like morning/afternoon/evening/night.>

Optionally, if you updated state pages:

PLAYER_STATUS:
<json object>

WORLD_STATUS:
<json object>

STATUS <CharacterName>:
<json object>

LOCATION_STATUS <LocationName>:
<json object>

Optionally, if a character's sprite or visibility should change for the next scene:

SPRITE <CharacterName>:
<json object like {{"sprite": "hidden", "visible_to": ["Alice"]}}>

HIDDEN <CharacterName>:
<list of observer names, or true>

Optionally, if new permanent facts were established:

FACT: [statement]
TRUST: 0.0
SOURCE: [who established it]

Optionally, if you want to leave a helpful note for the next scene's orchestrator:

ORCHESTRATOR_NOTE:
<free-form note, e.g. unresolved threads, suspicious behavior, or reminders>
"""


def summarizer_user_prompt(
    current_scene: Scene,
    current_scene_considerations: str,
    next_scene_plot: str,
    next_scene_considerations: str,
    location_desc: str,
    transcript: str,
    scratch_section: str,
    roster: list[str],
    continuing: list[str],
    new_arrivals: list[str],
    mechanical_changelog: list[dict[str, Any]],
    player_status: dict[str, Any],
    world_time: str,
    current_character_status: dict[str, dict[str, Any]],
    narrative_state: dict[str, Any],
    next_player_name: str,
    next_narrator_name: str,
    previous_scene_characters: list[str],
    summarizer_considerations: str = "",
    history_context: str = "",
) -> str:
    """Build the summarizer user prompt.

    :param current_scene: The scene that just ended.
    :param current_scene_considerations: Considerations from the ended scene.
    :param next_scene_plot: Plot of the upcoming scene.
    :param next_scene_considerations: Considerations from the upcoming scene.
    :param location_desc: Current primary location description.
    :param transcript: Full transcript of the ended scene.
    :param scratch_section: Private scratchpads from the ending scene.
    :param roster: Character roster for the next scene.
    :param continuing: Characters continuing into the next scene.
    :param new_arrivals: Characters new to the next scene.
    :param mechanical_changelog: Mechanical state changes during the scene.
    :param player_status: Player status at end of scene.
    :param world_time: World time at end of scene.
    :param current_character_status: Stored character status flags.
    :param narrative_state: Story-level narrative state.
    :param next_player_name: Player character name.
    :param next_narrator_name: Narrator name.
    :param previous_scene_characters: Characters in the ended scene.
    :param summarizer_considerations: Specific instructions for this transition.
    :param history_context: Relevant summaries from earlier scenes.
    :return: Formatted user prompt.
    """
    summarizer_note = ""
    if summarizer_considerations:
        summarizer_note = f"\nSpecific instructions for this transition:\n{summarizer_considerations}\n"
    changelog_block = "(No mechanical changes recorded.)"
    if mechanical_changelog:
        changelog_block = "\n".join(
            f"- turn {entry.get('turn', '?')}: {entry.get('type', 'change')} "
            f"{entry}"
            for entry in mechanical_changelog
        )

    player_status_block = "(No player status.)"
    if player_status:
        player_status_block = str(player_status)

    if current_character_status:
        status_lines = []
        for name, status in current_character_status.items():
            status_lines.append(f"- {name}: {status}")
        status_block = "\n".join(status_lines)
    else:
        status_block = "(No stored character statuses.)"

    narrative_state_block = "(No narrative state.)"
    if narrative_state:
        narrative_state_block = str(narrative_state)

    history_block = history_context if history_context else "(No relevant past-scene summaries.)"

    return f"""Current scene considerations:
{current_scene_considerations or "(none)"}

Upcoming scene plot:
{next_scene_plot}

Upcoming scene considerations:
{next_scene_considerations or "(none)"}
{summarizer_note}
Previous scene characters: {previous_scene_characters}
Next scene cast: {roster}
Characters continuing into next scene: {continuing}
Characters new to next scene: {new_arrivals}
Player character: {next_player_name or "(none)"}
Narrator: {next_narrator_name or "(none)"}

Current primary location description:
{location_desc}

World time at end of scene: {world_time or current_scene.time or "unspecified"}

Relevant summaries from earlier scenes:
{history_block}

Transcript of the scene that just ended:
{transcript}

Private scratchpads from the ending scene (use these to respect secrets):
{scratch_section}

Mechanical state changes during this scene:
{changelog_block}

Player player status at end of scene:
{player_status_block}

Stored character status flags (big off-screen events only):
{status_block}

Current story-level narrative state:
{narrative_state_block}
"""
