"""Scene-transition summarizer that bridges context between scenes.

The Summarizer runs during scene finalisation. It reads the full conversation
history of the ending scene plus the plot of the upcoming scene, and produces:

1. **Per-character bridging summaries** - one short narrative recap per
   character entering the next scene.  Characters who were *not* present in
   the scene that just ended receive a fuller recap of what they need to know.
   Characters who *were* present receive a minimal bridging note (e.g. time
   jumps).  The summarizer has access to the scratchpads of characters from the
   ending scene so that secrets and hidden agendas are respected.

2. A **finalized location description** - a coherent rewrite of the current
   location description that incorporates only *major* edits made by the
   orchestrator (e.g. "the house burned down").  Minor incidental edits
   (e.g. "a dog walked in") are discarded.

3. **Prefetched wiki context** - keyword search queries extracted from the
   upcoming scene's plot and considerations, run against the orchestrator wiki,
   and injected into the orchestrator prompt for the next scene.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ara.llm.context import ConversationContext
from ara.prompts.summarizer import summarizer_system_prompt, summarizer_user_prompt
from ara.llm.models import GameRole
from ara.utils.logger import get_logger

if TYPE_CHECKING:
    from ara.llm.client import LLMClient
    from ara.world.scene import Scene

logger = get_logger(__name__)

_BLOCK_HEADER_RE = re.compile(r"^([A-Z][A-Z0-9_]*(?:\s+\S+)?)\s*:(.*)$")


def _parse_json_or_string(text: str) -> Any:
    """Try to parse *text* as JSON; otherwise return the stripped string."""
    text = text.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return text


def _flush_modifier_block(
    block: tuple[str, str, list[str]],
    modifiers: SceneStateModifiers,
) -> None:
    """Parse a collected modifier block and store it in *modifiers*."""
    kind, name, lines = block
    raw = "\n".join(lines).strip()
    value = _parse_json_or_string(raw)
    if kind == "player_status" and isinstance(value, dict):
        modifiers.player_status = value
    elif kind == "world_status" and isinstance(value, dict):
        modifiers.world_status = value
    elif kind == "location_status" and isinstance(value, dict):
        modifiers.location_status[name] = value
    elif kind == "character_status" and isinstance(value, dict):
        modifiers.character_status[name] = value
    elif kind == "sprite":
        if isinstance(value, dict):
            modifiers.sprites[name] = value
        elif isinstance(value, str):
            modifiers.sprites[name] = {"sprite": value}
    elif kind == "hidden":
        entry: dict[str, Any] = {"sprite": "hidden"}
        if isinstance(value, list):
            entry["visible_to"] = list(value)
        elif isinstance(value, bool):
            entry["visible_to"] = []
        modifiers.sprites[name] = entry


def _extract_state_modifiers(text: str) -> tuple[str, SceneStateModifiers]:
    """Strip state-modifier blocks from *text* and return (cleaned_text, modifiers).

    Recognised modifier headers:
      PLAYER_STATUS:
      WORLD_STATUS:
      CHARACTER_STATUS <Name>:
      LOCATION_STATUS <Name>:
      SPRITE <Name>:
      HIDDEN <Name>:
    """
    # Strip markdown code-fence wrapping (some models wrap JSON in ```json ... ```).
    stripped_text = text.strip()
    if '\n' in stripped_text:
        first = stripped_text[:stripped_text.index('\n')].strip()
        if first in ('```json', '```') and stripped_text.endswith('```'):
            stripped_text = stripped_text[stripped_text.index('\n') + 1:-3].strip()

    modifiers = SceneStateModifiers()
    output_lines: list[str] = []
    current_block: tuple[str, str, list[str]] | None = None

    for line in stripped_text.splitlines():
        stripped = line.strip()
        m = _BLOCK_HEADER_RE.match(stripped) if stripped else None
        if m:
            # Flush any active modifier block before inspecting the new header.
            if current_block is not None:
                _flush_modifier_block(current_block, modifiers)
                current_block = None

            header = m.group(1)
            rest = m.group(2)
            upper = header.upper()
            # Determine if this is a state-modifier header and extract the target name.
            if upper.startswith("PLAYER_STATUS ") or upper == "PLAYER_STATUS":
                current_block = ("player_status", "", [rest] if rest else [])
                continue
            if upper.startswith("WORLD_STATUS ") or upper == "WORLD_STATUS":
                current_block = ("world_status", "", [rest] if rest else [])
                continue
            if upper.startswith("CHARACTER_STATUS "):
                name = header[len("CHARACTER_STATUS"):].strip()
                current_block = ("character_status", name, [rest] if rest else [])
                continue
            if upper.startswith("LOCATION_STATUS "):
                name = header[len("LOCATION_STATUS"):].strip()
                current_block = ("location_status", name, [rest] if rest else [])
                continue
            if upper.startswith("SPRITE "):
                name = header[len("SPRITE"):].strip()
                current_block = ("sprite", name, [rest] if rest else [])
                continue
            if upper.startswith("HIDDEN "):
                name = header[len("HIDDEN"):].strip()
                current_block = ("hidden", name, [rest] if rest else [])
                continue

        if current_block is not None:
            current_block[2].append(line)
        else:
            output_lines.append(line)

    if current_block is not None:
        _flush_modifier_block(current_block, modifiers)

    return "\n".join(output_lines), modifiers


@dataclass
class SceneStateModifiers:
    """Mechanical state changes produced by the summarizer for a scene."""

    player_status: dict[str, Any] = field(default_factory=dict)
    world_status: dict[str, Any] = field(default_factory=dict)
    character_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    location_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    sprites: dict[str, dict[str, Any]] = field(default_factory=dict)
    narrative_state: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(
            self.player_status
            or self.world_status
            or self.character_status
            or self.location_status
            or self.sprites
            or self.narrative_state
        )


class Summarizer:
    """Generates inter-scene bridging summaries and finalizes location state."""

    def __init__(self, client: LLMClient) -> None:
        """Create a summarizer.

        :param client: LLM client used for summarization calls.
        """
        self.client = client

    def summarize_transition(
        self,
        current_scene: Scene,
        current_scene_considerations: str,
        next_scene_plot: str,
        next_scene_considerations: str,
        conversation_context: list[dict],
        location_desc: str,
        language: str,
        scratchpads: dict[str, str],
        next_scene_chars: list[str],
        location_descs: dict[str, str] | None = None,
        next_scene_locations: list[str] | None = None,
        mechanical_changelog: list[dict[str, Any]] | None = None,
        player_status: dict[str, Any] | None = None,
        world_time: str = "",
        query_characters_fn: Callable[[str], list[str]] | None = None,
        next_scene_cast: list[str] | None = None,
        previous_scene_characters: list[str] | None = None,
        current_character_status: dict[str, dict[str, Any]] | None = None,
        narrative_state: dict[str, Any] | None = None,
        next_player_name: str = "",
        next_narrator_name: str = "",
        summarizer_considerations: str = "",
        history_context: str = "",
    ) -> tuple[dict[str, str], dict[str, str], str, list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], SceneStateModifiers, dict[str, dict[str, str]], dict[str, dict[str, str]], str]:
        """Produce per-character bridging summaries and finalized location descriptions.

        :param current_scene: The scene that just ended.
        :param current_scene_considerations: Considerations text of the scene that just ended.
        :param next_scene_plot: Plot text of the upcoming scene.
        :param next_scene_considerations: Considerations text of the upcoming scene.
        :param conversation_context: Orchestrator's curated view of the conversation
            history (observer ``__orchestrator__``, ``collapse=False``).
        :param location_desc: Current primary location description (may contain ``[Update]`` tags).
        :param language: Target language for output.
        :param scratchpads: Mapping from character name → scratch text.
        :param next_scene_chars: Names of all characters who will appear in the next scene.
        :param location_descs: Mapping from location name → current description for
            all locations present in the current scene.
        :param next_scene_locations: Names of locations that will appear in the next scene.
        :param mechanical_changelog: Mechanical state changes applied during the scene.
        :param player_status: Final player system-page state at the end of the scene.
        :param world_time: Final world time at the end of the scene.
        :param query_characters_fn: Optional callback to expand the character roster.
        :param next_scene_cast: Authoritative list of character names the summarizer
            must account for in the next scene. Defaults to ``next_scene_chars``.
        :param previous_scene_characters: List of characters present at the end of the
            previous scene, with anonymous members marked ``[anonymous]``.
        :param next_player_name: Display name of the upcoming scene's player character.
        :param next_narrator_name: Display name of the upcoming scene's narrator.
        :param current_character_status: Current stored status flags for all known characters.
        :param narrative_state: Current story-level narrative state flags.
        :param summarizer_considerations: Author-supplied instructions for how to
            summarize this specific transition (from ``[plot.next.<id>].summarizer_considerations``).
        :param history_context: Relevant past-scene summaries retrieved from the
            long-term story memory, if any.
        :return: ``(per_character_summaries, finalized_location_descs, time, facts, player_status_delta, character_status_updates, narrative_state, state_modifiers, character_overrides, anonymous_chars, orchestrator_note)``.
        """
        location_descs = dict(location_descs) if location_descs else {}
        next_scene_locations = list(next_scene_locations) if next_scene_locations else []
        mechanical_changelog = list(mechanical_changelog) if mechanical_changelog else []
        player_status = dict(player_status) if player_status else {}
        next_scene_cast = list(next_scene_cast) if next_scene_cast else list(next_scene_chars)
        previous_scene_characters = list(previous_scene_characters) if previous_scene_characters else []
        current_character_status = dict(current_character_status) if current_character_status else {}
        narrative_state = dict(narrative_state) if narrative_state else {}

        # Build a compact transcript from the already-curated conversation context.
        # The caller is expected to pass the orchestrator's view, where the
        # orchestrator is the only assistant and every other speaker is reported
        # through user messages.
        transcript = ConversationContext.to_narrative_text(
            conversation_context,
            observer_name="Orchestrator",
            max_lines=50,
        )

        # Build scratchpad section.
        scratch_lines: list[str] = []
        for name, text in scratchpads.items():
            if text and text != "Nothing yet!":
                scratch_lines.append(f"--- {name}'s scratchpad ---\n{text}")
        scratch_section = "\n\n".join(scratch_lines) if scratch_lines else "(No scratchpads available.)"

        # Determine which characters are continuing vs. new.
        prev_char_names = {c.name for c in current_scene.character_pool}

        def _run(roster: list[str]) -> tuple[str, dict[str, Any]]:
            continuing = [c for c in roster if c in prev_char_names]
            new_arrivals = [c for c in roster if c not in prev_char_names]

            system_prompt = self._build_system_prompt(
                language=language,
                has_changelog=bool(mechanical_changelog),
                has_player_status=bool(player_status),
            )
            user_prompt = self._build_user_prompt(
                current_scene=current_scene,
                current_scene_considerations=current_scene_considerations,
                next_scene_plot=next_scene_plot,
                next_scene_considerations=next_scene_considerations,
                location_desc=location_desc,
                transcript=transcript,
                scratch_section=scratch_section,
                roster=roster,
                continuing=continuing,
                new_arrivals=new_arrivals,
                mechanical_changelog=mechanical_changelog,
                player_status=player_status,
                world_time=world_time,
                current_character_status=current_character_status,
                narrative_state=narrative_state,
                next_player_name=next_player_name,
                next_narrator_name=next_narrator_name,
                previous_scene_characters=previous_scene_characters,
                summarizer_considerations=summarizer_considerations,
                history_context=history_context,
            )

            result = self.client.complete(
                role=GameRole.SUMMARIZER,
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                stream=False,
            )

            text = result.content.strip()
            return text, player_status

        text, _ = _run(next_scene_cast)

        # Optional one-time roster expansion via query_characters tool.
        if query_characters_fn is not None:
            query = self._extract_query(text)
            if query:
                extra = query_characters_fn(query)
                if extra:
                    expanded_roster = list(dict.fromkeys(next_scene_cast + [line.split(":")[0].strip() for line in extra if ":" in line]))
                    text, _ = _run(expanded_roster)

        summaries, primary_location, time, facts, player_status_delta, character_status_updates, narrative_state, state_modifiers, character_overrides, anonymous_chars, orchestrator_note = self._parse_response(
            text, next_scene_chars, location_desc, current_scene.time
        )

        logger.debug(
            f"Summarizer raw response ({len(text)} chars):\n{text[:2000]}"
        )
        logger.debug(
            f"Parsed summaries: {summaries}\n"
            f"Facts: {facts}\n"
            f"Location: {primary_location}\n"
            f"Time: {time}"
        )

        # Determine which locations may need rewriting beyond the primary one.
        relevant_locations = {
            name: desc
            for name, desc in location_descs.items()
            if name in next_scene_locations or name == current_scene.starting_location.name
        }

        finalized_descs = self._finalize_locations(
            relevant_locations=relevant_locations,
            primary_name=current_scene.starting_location.name,
            primary_desc=primary_location,
            transcript=transcript,
            language=language,
        )

        return summaries, finalized_descs, time, facts, player_status_delta, character_status_updates, narrative_state, state_modifiers, character_overrides, anonymous_chars, orchestrator_note

    def prefetch_wiki_context(
        self,
        plot: str,
        considerations: str,
        world: str,
        zeitgeist: str,
        tone: str,
        language: str,
        wiki_recall_fn: Callable[..., str],
        max_distance: float | None = 0.65,
    ) -> str:
        """Generate keyword queries from the upcoming scene and search the wiki.

        The keywords are extracted by a cheap subagent call, then each query is
        run through *wiki_recall_fn*.  Results are deduplicated and returned as
        a single formatted block ready to be injected into the orchestrator
        prompt.

        :param plot: Upcoming scene plot text.
        :param considerations: Upcoming scene considerations.
        :param world: Current world identifier.
        :param zeitgeist: Current scene zeitgeist.
        :param tone: Current scene tone.
        :param language: Target language for keyword extraction.
        :param wiki_recall_fn: Callable that takes a query string and optional
            ``exclude_docs`` / ``max_distance`` kwargs and returns formatted
            wiki results.
        :param max_distance: Maximum ChromaDB distance for a result to be
            considered relevant.  Default 0.65.
        :return: Combined wiki context block, or an empty string if nothing
            relevant was found.
        """
        prompt = f"""IMPORTANT: Respond in {language} only.
Given the upcoming scene details below, produce 3-5 short keyword search queries
for a vector wiki of world facts. Each query should be a short phrase likely to
retrieve relevant setting, lore, or plot background. Return ONLY one query per
line, with no numbering, headers, or prose.

Upcoming scene plot:
{plot}

Upcoming scene considerations:
{considerations}

World: {world}
Zeitgeist: {zeitgeist}
Tone: {tone}
"""
        try:
            raw = self.client.complete_subagent(
                task="Extract wiki search keywords for the upcoming scene.",
                context=prompt,
                max_tokens=128,
            )
        except Exception as exc:
            logger.debug(f"Wiki keyword extraction failed: {exc}")
            return ""

        queries = [line.strip() for line in raw.splitlines() if line.strip()]
        logger.debug(f"Wiki prefetch queries: {queries}")
        if not queries:
            return ""

        seen_docs: set[str] = set()
        chunks: list[str] = []
        for query in queries:
            try:
                result = wiki_recall_fn(
                    query, exclude_docs=seen_docs, max_distance=max_distance
                )
            except Exception as exc:
                logger.debug(f"Wiki recall failed for query '{query}': {exc}")
                continue
            if (
                not result
                or "No relevant" in result
                or "nothing relevant" in result.lower()
                or "already covered" in result.lower()
            ):
                continue
            new_docs: list[str] = []
            for doc in result.split("\n\n"):
                doc = doc.strip()
                if not doc.startswith("-"):
                    continue
                normalized = doc.lstrip("-").strip()
                if normalized.startswith("(trust:") and ")" in normalized:
                    normalized = normalized.split(")", 1)[1].strip()
                normalized = " ".join(normalized.split())
                if normalized and normalized not in seen_docs:
                    seen_docs.add(normalized)
                    new_docs.append(doc)
            if new_docs:
                chunks.extend(new_docs)
                logger.debug(
                    f"Wiki prefetch query '{query}' added {len(new_docs)} new document(s)."
                )
        combined = "\n\n".join(chunks)
        logger.debug(
            f"Wiki prefetch combined context ({len(chunks)} documents):\n{combined}"
        )
        return combined

    def apply_initial_state_modifiers(
        self,
        scene: Scene,
        language: str,
    ) -> SceneStateModifiers:
        """Produce initial state modifiers for the first scene of a story.

        Reads the scene's plot and considerations and emits any mechanical
        state (hidden characters, status pages, sprite overrides) that should
        be in place before the first turn.
        """
        prompt = f"""IMPORTANT: Respond in {language} only.
You are the State Modifier. Read the upcoming scene below and emit any initial mechanical state that should be applied before the first turn.

Allowed blocks:
- PLAYER_STATUS: <system-page DSL json>
- WORLD_STATUS: <system-page DSL json>
- CHARACTER_STATUS <CharacterName>: <system-page DSL json>
- LOCATION_STATUS <LocationName>: <system-page DSL json>
- SPRITE <CharacterName>: {{"sprite": "hidden", "visible_to": ["ObserverName"]}}
- HIDDEN <CharacterName>: ["ObserverName"] or true

Only emit a block if the scene explicitly requires it. If nothing needs to change, emit nothing.

Scene plot:
{scene.plot_story}

Scene considerations:
{scene.plot_considerations or "(none)"}
"""
        try:
            raw = self.client.complete_subagent(
                task="Apply initial state modifiers for the first scene.",
                context=prompt,
                max_tokens=512,
            )
        except Exception as exc:
            logger.warning(f"Initial state modifier summarizer failed: {exc}")
            return SceneStateModifiers()

        _, modifiers = _extract_state_modifiers(raw.strip())
        return modifiers

    @staticmethod
    def _extract_query(text: str) -> str | None:
        """Return the first QUERY: line if the response is only query lines."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        if not all(line.upper().startswith("QUERY:") for line in lines):
            return None
        return lines[0][len("QUERY:"):].strip()

    @staticmethod
    def _build_system_prompt(
        language: str,
        has_changelog: bool,
        has_player_status: bool,
    ) -> str:
        return summarizer_system_prompt(language, has_changelog, has_player_status)

    @staticmethod
    def _build_user_prompt(
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
        return summarizer_user_prompt(
            current_scene,
            current_scene_considerations,
            next_scene_plot,
            next_scene_considerations,
            location_desc,
            transcript,
            scratch_section,
            roster,
            continuing,
            new_arrivals,
            mechanical_changelog,
            player_status,
            world_time,
            current_character_status,
            narrative_state,
            next_player_name,
            next_narrator_name,
            previous_scene_characters,
            summarizer_considerations,
            history_context,
        )

    def _finalize_locations(
        self,
        relevant_locations: dict[str, str],
        primary_name: str,
        primary_desc: str,
        transcript: str,
        language: str,
    ) -> dict[str, str]:
        """Return updated descriptions for all relevant locations.

        Uses a classifier subagent to decide whether secondary locations changed.
        Only locations marked as ``changed`` trigger an extra LLM call.
        """
        finalized: dict[str, str] = dict(relevant_locations)
        finalized[primary_name] = primary_desc

        secondary = {n: d for n, d in relevant_locations.items() if n != primary_name}
        if not secondary:
            return finalized

        try:
            changed = self._classify_locations(secondary, transcript, language)
        except Exception as exc:
            logger.debug(f"Location classification failed: {exc}")
            return finalized

        for name in changed:
            if name not in secondary:
                continue
            try:
                new_desc = self._rewrite_location(
                    name, secondary[name], transcript, language
                )
                if new_desc:
                    finalized[name] = new_desc
            except Exception as exc:
                logger.debug(f"Location rewrite failed for {name}: {exc}")

        return finalized

    def _classify_locations(
        self,
        locations: dict[str, str],
        transcript: str,
        language: str,
    ) -> list[str]:
        """Ask a subagent which secondary locations changed materially.

        :return: List of location names flagged as ``changed``.
        """
        location_block = "\n\n".join(
            f"{name}:\n{desc}" for name, desc in locations.items()
        )
        prompt = f"""IMPORTANT: Respond in {language} only.
You are a classifier. Given the scene transcript and a list of locations, decide
whether each location changed in a major, lasting way during the scene.

Locations:
{location_block}

Transcript:
{transcript}

Return one line per location in this exact format:
location_name: NOOP
location_name: changed (brief reason)

Only mark a location as changed if something major happened (e.g. it burned
down, was ransacked, a bridge collapsed). Minor changes should be NOOP.
"""
        result = self.client.complete_subagent(
            task="Classify which locations changed materially.",
            context=prompt,
            max_tokens=256,
        )
        changed: list[str] = []
        for line in result.splitlines():
            if ":" not in line:
                continue
            name, verdict = line.split(":", 1)
            name = name.strip()
            verdict = verdict.strip().lower()
            if name in locations and not verdict.startswith("noop"):
                changed.append(name)
        return changed

    def _rewrite_location(
        self,
        name: str,
        current_desc: str,
        transcript: str,
        language: str,
    ) -> str:
        """Use a subagent to rewrite a single location description."""
        prompt = f"""IMPORTANT: Respond in {language} only.
Rewrite the description of '{name}' so any major permanent changes from the
scene are incorporated smoothly. Ignore minor incidental changes. If nothing
major changed, return the original description cleaned of any [Update] tags.

Current description:
{current_desc}

Scene transcript:
{transcript}

Return ONLY the new description, with no extra headers.
"""
        return self.client.complete_subagent(
            task=f"Rewrite the description for {name}.",
            context=prompt,
            max_tokens=256,
        ).strip()

    @staticmethod
    def _parse_response(
        text: str,
        next_scene_chars: list[str],
        fallback_location: str,
        fallback_time: str = "",
    ) -> tuple[dict[str, str], str, str, list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], SceneStateModifiers, dict[str, dict[str, str]], dict[str, dict[str, str]], str]:
        """Parse the summarizer output format.

        Recognised blocks: SUMMARY, LOCATION, TIME, FACT, PLAYER_STATUS,
        WORLD_STATUS, STATUS <Name>, LOCATION_STATUS <Name>, SPRITE <Name>,
        HIDDEN <Name>, NARRATIVE_STATE, CHARACTER <Name>, ANONYMOUS <Name>,
        ORCHESTRATOR_NOTE.
        """
        text, modifiers = _extract_state_modifiers(text)
        summaries: dict[str, str] = {}
        location = fallback_location
        time = fallback_time
        time_found: bool = False
        facts: list[dict[str, Any]] = []
        player_status_delta: dict[str, Any] = {}
        character_status_updates: dict[str, dict[str, Any]] = {}
        narrative_state: dict[str, Any] = {}
        character_overrides: dict[str, dict[str, str]] = {}
        anonymous_chars: dict[str, dict[str, str]] = {}
        orchestrator_note: str = ""

        section: str | None = None
        buffer: list[str] = []
        status_buffer: list[str] = []
        narrative_state_buffer: list[str] = []
        override_buffer: list[str] = []
        anon_buffer: list[str] = []
        note_buffer: list[str] = []
        current_char: str | None = None
        current_status_char: str | None = None
        current_override_char: str | None = None
        current_anon_char: str | None = None
        current_fact: dict[str, Any] | None = None

        def _flush_char() -> None:
            nonlocal current_char, buffer
            if current_char is not None and buffer:
                summaries[current_char] = "\n".join(buffer).strip()
                buffer = []
            current_char = None

        def _flush_section() -> None:
            nonlocal section, buffer, location, time, time_found
            joined = "\n".join(buffer).strip()
            if section == "location":
                if joined.lower() in {"morning", "afternoon", "evening", "night", "dawn", "dusk", "day", "midnight"}:
                    time = joined
                    time_found = True
                else:
                    location = joined
            elif section == "time":
                if joined.lower() in {"morning", "afternoon", "evening", "night", "dawn", "dusk", "day", "midnight"}:
                    time = joined
                    time_found = True
                else:
                    location = joined
            elif section is None and joined:
                # Unheaded trailing text: treat as location unless it looks like a time word.
                if joined.lower() in {"morning", "afternoon", "evening", "night", "dawn", "dusk", "day", "midnight"}:
                    time = joined
                    time_found = True
                else:
                    location = joined
            section = None
            buffer = []

        def _start_fact() -> None:
            nonlocal current_fact
            if current_fact is not None and current_fact.get("fact"):
                facts.append(current_fact)
            current_fact = {"trust": 0.0, "source": ""}

        def _flush_fact() -> None:
            nonlocal current_fact
            if current_fact is not None and current_fact.get("fact"):
                current_fact["fact"] = current_fact["fact"].strip()
                facts.append(current_fact)
            current_fact = None

        def _flush_status() -> None:
            nonlocal current_status_char, status_buffer
            if current_status_char is not None and status_buffer:
                joined = "\n".join(status_buffer).strip()
                try:
                    character_status_updates[current_status_char] = json.loads(joined)
                except Exception:
                    logger.debug(f"Could not parse STATUS block for {current_status_char}: {joined[:200]}")
            current_status_char = None
            status_buffer = []

        def _flush_override() -> None:
            nonlocal current_override_char, override_buffer
            if current_override_char is not None and override_buffer:
                joined = "\n".join(override_buffer).strip()
                try:
                    parsed = json.loads(joined)
                    if isinstance(parsed, dict):
                        character_overrides[current_override_char] = {
                            k: v for k, v in parsed.items() if isinstance(v, str) and v.strip()
                        }
                except Exception:
                    logger.debug(f"Could not parse CHARACTER block for {current_override_char}: {joined[:200]}")
            current_override_char = None
            override_buffer = []

        def _flush_anon() -> None:
            nonlocal current_anon_char, anon_buffer
            if current_anon_char is not None and anon_buffer:
                joined = "\n".join(anon_buffer).strip()
                try:
                    parsed = json.loads(joined)
                    if isinstance(parsed, dict):
                        anonymous_chars[current_anon_char] = {
                            k: v for k, v in parsed.items() if isinstance(v, str)
                        }
                    else:
                        anonymous_chars[current_anon_char] = {"description": str(parsed)}
                except Exception:
                    anonymous_chars[current_anon_char] = {"description": joined}
            current_anon_char = None
            anon_buffer = []

        for raw_line in text.splitlines():
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            upper = stripped.upper()

            # SUMMARY <Name>:
            if upper.startswith("SUMMARY ") and ":" in stripped:
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                name_part = stripped[len("SUMMARY "):stripped.rfind(":")].strip()
                current_char = name_part
                section = "char"
                continue

            # STATUS <Name>:
            if upper.startswith("STATUS ") and ":" in stripped:
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                name_part = stripped[len("STATUS "):stripped.rfind(":")].strip()
                current_status_char = name_part
                section = "status"
                continue

            # LOCATION:
            if upper == "LOCATION:":
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                section = "location"
                continue

            # TIME:
            if upper == "TIME:":
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                section = "time"
                continue

            # TIME: <value>  (inline format)
            if upper.startswith("TIME:") and len(stripped) > 5:
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                time = stripped[stripped.find(":") + 1:].strip()
                time_found = True
                section = None
                continue

            # NARRATIVE_STATE:
            if upper == "NARRATIVE_STATE:":
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                section = "narrative_state"
                continue

            # ORCHESTRATOR_NOTE:
            if upper == "ORCHESTRATOR_NOTE:":
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                section = "orchestrator_note"
                continue

            # FACT: (may have content on the same line)
            if upper.startswith("FACT") and ":" in stripped:
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                section = "fact"
                _start_fact()
                fact_text = stripped[stripped.find(":") + 1:].strip()
                if fact_text:
                    current_fact["fact"] = fact_text.strip()
                continue

            # TRUST: (may have value on the same line)
            if upper.startswith("TRUST") and current_fact is not None:
                value = stripped[stripped.find(":") + 1:].strip()
                try:
                    current_fact["trust"] = float(value or "0")
                except ValueError:
                    current_fact["trust"] = 0.0
                continue

            # SOURCE: (may have value on the same line)
            if upper.startswith("SOURCE") and current_fact is not None:
                current_fact["source"] = stripped[stripped.find(":") + 1:].strip()
                continue

            # CHARACTER <Name>:
            if upper.startswith("CHARACTER ") and ":" in stripped:
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                name_part = stripped[len("CHARACTER "):stripped.rfind(":")].strip()
                current_override_char = name_part
                section = "override"
                continue

            # ANONYMOUS <Name>:
            if upper.startswith("ANONYMOUS ") and ":" in stripped:
                _flush_char()
                _flush_section()
                _flush_fact()
                _flush_status()
                _flush_override()
                _flush_anon()
                name_part = stripped[len("ANONYMOUS "):stripped.rfind(":")].strip()
                current_anon_char = name_part
                section = "anonymous"
                continue

            # Accumulate content
            if section == "fact" and current_fact is not None:
                if "fact" not in current_fact:
                    if stripped:
                        current_fact["fact"] = stripped
                else:
                    current_fact["fact"] += "\n" + stripped
            elif section == "status":
                if stripped:
                    status_buffer.append(line)
            elif section == "override":
                if stripped:
                    override_buffer.append(line)
            elif section == "anonymous":
                if stripped:
                    anon_buffer.append(line)
            elif section == "narrative_state":
                if stripped:
                    narrative_state_buffer.append(line)
            elif section == "orchestrator_note":
                if stripped:
                    note_buffer.append(line)
            elif section is not None:
                buffer.append(line)
            else:
                # Collect unheaded lines so the final flush can treat them as a location fallback.
                buffer.append(line)

        # Final flush
        _flush_char()
        _flush_section()
        _flush_fact()
        _flush_status()
        _flush_override()
        _flush_anon()


        # Parse any NARRATIVE_STATE block we collected.
        if narrative_state_buffer:
            joined_narrative = "\n".join(narrative_state_buffer).strip()
            try:
                narrative_state = json.loads(joined_narrative)
            except Exception:
                logger.debug(f"Could not parse NARRATIVE_STATE block: {joined_narrative[:200]}")

        # Collect orchestrator_note block.
        if note_buffer:
            orchestrator_note = "\n".join(note_buffer).strip()

        if not player_status_delta and modifiers.player_status:
            player_status_delta = dict(modifiers.player_status)

        # Fallback: if parsing completely failed, treat whole text as a generic
        # summary assigned to every character.
        if not summaries and not location and not facts and not character_status_updates and not narrative_state and not character_overrides and not modifiers.player_status and not modifiers.world_status and not modifiers.location_status and not modifiers.sprites:
            for name in next_scene_chars:
                summaries[name] = text
            location = fallback_location

        # Ensure every expected character has at least an empty summary.
        for name in next_scene_chars:
            if name not in summaries:
                summaries[name] = ""

        return summaries, location, time if time_found else "", facts, player_status_delta, character_status_updates, narrative_state, modifiers, character_overrides, anonymous_chars, orchestrator_note
