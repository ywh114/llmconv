"""Scene-transition summarizer that bridges context between scenes.

The Summarizer runs during scene finalisation. It reads the full conversation
history of the ending scene plus the plot of the upcoming scene, and produces:

1. **Per-character bridging summaries** — one short narrative recap per
   character entering the next scene.  Characters who were *not* present in the
   scene that just ended receive a fuller recap of what they need to know.
   Characters who *were* present receive a minimal bridging note (e.g. time
   jumps).  The summarizer has access to the scratchpads of characters from the
   ending scene so that secrets and hidden agendas are respected.

2. A **finalized location description** — a coherent rewrite of the current
   location description that incorporates only *major* edits made by the
   orchestrator (e.g. "the house burned down").  Minor incidental edits
   (e.g. "a dog walked in") are discarded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ara.models import GameRole
from ara.utils.logger import get_logger

if TYPE_CHECKING:
    from ara.llm.client import LLMClient
    from ara.world.scene import Scene

logger = get_logger(__name__)


class Summarizer:
    """Generates inter-scene bridging summaries and finalizes location state."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def summarize_transition(
        self,
        current_scene: Scene,
        next_scene_plot: str,
        next_scene_considerations: str,
        conversation_context: list[dict],
        location_desc: str,
        language: str,
        scratchpads: dict[str, str],
        next_scene_chars: list[str],
    ) -> tuple[dict[str, str], str]:
        """Produce per-character bridging summaries and a finalized location description.

        :param current_scene: The scene that just ended.
        :param next_scene_plot: Plot text of the upcoming scene.
        :param next_scene_considerations: Considerations text of the upcoming scene.
        :param conversation_context: Full conversation history as message dicts.
        :param location_desc: Current location description (may contain ``[Update]`` tags).
        :param language: Target language for output.
        :param scratchpads: Mapping from character name → scratch text for every
            character who participated in *current_scene*.  These are the
            characters' private notes; use them to respect secrets and hidden
            agendas when deciding what each new arrival should know.
        :param next_scene_chars: Names of all characters who will appear in the
            next scene (including the player and narrator).
        :return: ``(per_character_summaries, finalized_location_desc)`` where
            *per_character_summaries* maps each character name from
            *next_scene_chars* to their tailored summary string.
        """
        # Build a compact transcript from the conversation context.
        transcript_lines: list[str] = []
        for msg in conversation_context:
            role = msg.get("role", "")
            name = msg.get("name", "")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "user" and name:
                transcript_lines.append(f"{name}: {content}")
            elif role == "assistant" and name:
                transcript_lines.append(f"{name}: {content}")
            elif role == "system":
                transcript_lines.append(f"[System]: {content}")

        transcript = "\n".join(transcript_lines[-50:])  # last 50 messages max

        # Build scratchpad section.
        scratch_lines: list[str] = []
        for name, text in scratchpads.items():
            if text and text != "Nothing yet!":
                scratch_lines.append(f"--- {name}'s scratchpad ---\n{text}")
        scratch_section = "\n\n".join(scratch_lines) if scratch_lines else "(No scratchpads available.)"

        # Determine which characters are continuing vs. new.
        prev_char_names = {c.name for c in current_scene.character_pool}
        continuing = [c for c in next_scene_chars if c in prev_char_names]
        new_arrivals = [c for c in next_scene_chars if c not in prev_char_names]

        system_prompt = f"""IMPORTANT: Write in {language} only!
You are the Summarizer — a background agent that prepares narrative continuity
between scenes in a visual novel.

## Task
Read the transcript of the scene that just ended, the private scratchpads of
the characters who were in that scene, and the plot of the upcoming scene.
Produce ONE summary for EACH character who will appear in the next scene, plus
a finalized location description.

### Rules for summaries
1. **Characters who were already present** (continuing characters) receive a
   very short bridging note — just enough to cover time jumps or emotional
   residue. 1–2 sentences max.
2. **Characters who are NEW to the next scene** receive a fuller recap of the
   key facts, emotional state, and unresolved threads they would reasonably
   know about. 2–4 sentences.
3. **Respect secrets.** Use the scratchpads to know what each character secretly
   knows or plans.  Do NOT reveal a secret in a summary meant for a character
   who should not know it.
4. Write each summary as neutral narration from that character's point of view,
   NOT as dialogue.  Do NOT use meta-language like "the player" or "the scene".

### Finalized location description
Rewrite the current location description so any major permanent changes are
incorporated smoothly.  ONLY keep changes that alter the location in a lasting
way (e.g. a house burned down, a bridge collapsed).  Ignore minor incidental
changes (e.g. a dog walked in, someone moved a chair).  If no major changes
occurred, return the original description cleaned of any ``[Update]`` tags.

Return your response in this exact format:

SUMMARY <CharacterName>:
<summary for this character>

(repeat for every character in the next scene)

LOCATION:
<finalized location description>
"""

        user_prompt = f"""Upcoming scene plot:
{next_scene_plot}

Upcoming scene considerations:
{next_scene_considerations}

Characters continuing into next scene: {continuing}
Characters new to next scene: {new_arrivals}

Current location description:
{location_desc}

Transcript of the scene that just ended:
{transcript}

Private scratchpads from the ending scene (use these to respect secrets):
{scratch_section}
"""

        result = self.client.complete(
            role=GameRole.SUMMARIZER,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            stream=False,
        )

        text = result.content.strip()
        return self._parse_response(text, next_scene_chars, location_desc)

    @staticmethod
    def _parse_response(
        text: str,
        next_scene_chars: list[str],
        fallback_location: str,
    ) -> tuple[dict[str, str], str]:
        """Parse the SUMMARY <Name>: / LOCATION format from the LLM response."""
        summaries: dict[str, str] = {}
        location = fallback_location

        lines = text.splitlines()
        current_char: str | None = None
        section_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            upper = stripped.upper()

            # Detect SUMMARY <Name>:
            if upper.startswith("SUMMARY ") and ":" in stripped:
                # Flush previous section
                if current_char is not None:
                    summaries[current_char] = "\n".join(section_lines).strip()
                elif section_lines:
                    # We were in LOCATION section
                    location = "\n".join(section_lines).strip()

                # Extract character name after "SUMMARY " and before ":"
                name_part = stripped[len("SUMMARY "):stripped.rfind(":")].strip()
                current_char = name_part
                section_lines = []
                continue

            # Detect LOCATION:
            if upper == "LOCATION:":
                if current_char is not None:
                    summaries[current_char] = "\n".join(section_lines).strip()
                current_char = None
                section_lines = []
                continue

            # Accumulate lines
            if current_char is not None or (not upper.startswith("SUMMARY") and upper != "LOCATION:"):
                section_lines.append(line)

        # Flush final section
        if current_char is not None:
            summaries[current_char] = "\n".join(section_lines).strip()
        elif section_lines:
            location = "\n".join(section_lines).strip()

        # Fallback: if parsing completely failed, treat whole text as a generic
        # summary assigned to every character.
        if not summaries and not location:
            for name in next_scene_chars:
                summaries[name] = text
            location = fallback_location

        # Ensure every expected character has at least an empty summary.
        for name in next_scene_chars:
            if name not in summaries:
                summaries[name] = ""

        return summaries, location
