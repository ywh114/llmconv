"""Orchestrator agent that decides who speaks next and how the scene advances."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.tools import ToolRegistry, tool
from ara.models import GameRole
from ara.world.character import Character
from ara.world.scene import Location, Scene
from ara.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TurnDecision:
    """Output of a single orchestrator turn.

    :ivar next_char: The character (or narrator) selected to act next.
    :ivar directive: In-universe instruction for the next speaker.  Empty
        when the next speaker is the player.
    :ivar suggestions: Options offered to the player.  Empty for NPC turns.
    :ivar entering_chars: Characters that should enter the scene at the start
        of this round.
    :ivar exiting_chars: Characters that should exit the scene at the end of
        this round.
    :ivar switch_location: New location to switch to, or ``None``.
    :ivar edit_location: Description of how the current location was modified,
        or empty string if no change.
    :ivar next_scene: If non-``None``, the scene ends and transitions to the
        named follow-up scene.
    :ivar change_sprite: Mapping of character names → sprite names to switch
        to this turn.
    """

    next_char: Character
    directive: str
    suggestions: list[str]
    entering_chars: set[Character]
    exiting_chars: set[Character]
    switch_location: Location | None
    edit_location: str = ""
    next_scene: str | None = None
    change_sprite: dict[str, str] = field(default_factory=dict)


class Orchestrator:
    """Builds dynamic tool schemas and calls the LLM to direct scene flow.

    Because the list of valid characters, locations, and scene choices changes
    every turn, the ``next_round`` tool schema is regenerated on each call.
    When strict mode is enabled, the schema includes ``strict: true`` so that
    DeepSeek's beta endpoint guarantees JSON-schema-compliant output,
    eliminating the need for fragile retry loops.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.registry = ToolRegistry()
        self._capture: _NextRoundCapture | None = None

    def _system_prompt(
        self, player: Character, narrator: Character, scene: Scene
    ) -> str:
        """Build the orchestrator system prompt.

        :param player: Player-controlled character.
        :param narrator: Narrator character.
        :param scene: Current scene.
        :return: Formatted system prompt string.
        """
        return f"""IMPORTANT: Give suggestions and directives in {scene.language} only!
# Role: Visual Novel Orchestrator
## Goal
You are the Orchestrator/DM for a visual novel, with the player taking assuming the role of {player.name}.
The narrator name is {narrator.name}.
The zeitgeist of the plot is: {scene.zeitgeist}.
The tone of the current scene is: {scene.tone}.

When the scene goes off-script, use directives and the narrator to force it back.
ALWAYS be pushing the plot forwards.
DO NOT add any extraneous events.

## ABSOLUTE RULES — DO NOT VIOLATE
- **NEVER roleplay. NEVER speak in character. NEVER output dialogue, narration, or prose.**
  Your ONLY output is a tool call. You are a machine that decides who speaks next.
- **NEVER act as the narrator.** The narrator is a separate character ({narrator.name}).
  You merely decide WHEN the narrator speaks; you do NOT write their lines.
- **You MUST call the `next_round` tool on EVERY single turn.**
  No exceptions. No free text. No thinking out loud. Just the tool call.

## Core Responsibilities
1. **Control Narrative Flow**:
   - Select next character after each dialogue turn (Character, Narrator).
   - Be proactive in using switch_location to switch between locations in the scene.
   - Use directives to guide characters through the scene's plot.
   - Use suggestions to guide players through the scene's plot towards one of the specified outcomes.
   - Choose what characters enter/exit the scene based on the scene's plot.

2. **Principled Guidance**
    - Directives must be in-universe: minimize meta-language.
    - Narrator control: use ONLY for environmental shifts and scene description. The Narrator should only write one or two sentences at a time. Do NOT use the narrator for back-to-back atmospheric padding.
    - End the scene IMMEDIATELY when the plot's conclusion is reached. Do NOT add a closing narration before ending.
    - When ending the scene, set `end_scene` to `true` and `next_scene` to the most appropriate follow-up scene.
    - When `end_scene` is `true`, `next_scene` MUST be a valid scene ID (not empty).
    - If the turn count is high (8+), wrap up the scene quickly and end it.

3. **Tool instructions**
    - You MUST use the `next_round` tool on EVERY turn. Do NOT output free text. Always call the tool.
    - Use the next_character field to specify the next character.
    - Use the directive field to provide directives to the next character, if it is not the player.
    - If the next character is the player, provide an array of suggestions that correspond to the possible outcomes of the scene.
    - Entering characters enter at the start of the current round of conversation. However, they CANNOT BE the next speaker.
    - Exiting characters exit at the end of the current round of conversation. They CAN BE the next speaker.
"""

    def decide_next_turn(
        self,
        scene: Scene,
        ctx: ConversationContext,
        here_chars: set[Character],
        away_chars: set[Character],
        prev_char: Character | None,
        loc: Location,
        history: str = "",
        turn_count: int = 0,
        speaker_history: list[str] | None = None,
        _retries: int = 3,
    ) -> TurnDecision:
        """Ask the LLM to choose the next speaker and scene adjustments.

        Retries automatically when the model fails to produce a valid tool call.

        :param scene: Current scene definition.
        :param ctx: Base conversation context.
        :param here_chars: Characters currently present.
        :param away_chars: Characters currently off-scene.
        :param prev_char: The character who spoke in the previous round, or
            ``None`` on the first turn.
        :param loc: Current location.
        :param _retries: Internal retry counter.
        :return: Parsed :class:`TurnDecision`.
        """
        self._capture = _NextRoundCapture()
        self.registry.register("next_round", self._capture.hook)

        list_here = [c.name for c in here_chars]
        list_away = [c.name for c in away_chars]
        prev_name = prev_char.name if prev_char else None

        speaker_history = speaker_history or []
        recent = speaker_history[-3:] if speaker_history else []

        # Build valid_next: exclude previous speaker to avoid back-to-back repeats.
        # The plot must explicitly call for back-to-back narration or monologue.
        valid_next = [c.name for c in here_chars if c != prev_char]

        # Narrator cooldown: exclude if they spoke in the last 2 turns,
        # unless the plot explicitly requires narration.
        narrator_spoke_recently = scene.narrator.name in recent[-2:]
        if scene.narrator != prev_char and not narrator_spoke_recently:
            valid_next.append(scene.narrator.name)

        # If everyone was excluded (edge case: single-character scene), fall back
        # to player + narrator.
        if not valid_next:
            valid_next = [scene.player.name, scene.narrator.name]

        valid_locs = [loc.name for loc in scene.location_pool]
        valid_scenes = list(scene.next_choices.keys())

        # Build a description of available sprites for the tool prompt.
        sprite_info_lines = []
        for c in here_chars:
            if c.sprites:
                sprite_info_lines.append(
                    f"  {c.name}: {', '.join(c.sprites)} (current: {c.current_sprite})"
                )
        sprite_info = "\n".join(sprite_info_lines) if sprite_info_lines else "  (none)"

        next_round_tool = tool(
            name="next_round",
            description="""Choose the next character/narrator to act, or to end the scene.
If choosing an NPC/the narrator, provide a directive to guide their actions to fulfill the given plot.
If choosing the player, provide suggestions.
If setting end_scene to True, you can leave everything else blank.
Also decide on what characters enter/exit the scene.
""",
            properties={
                "next_character": {
                    "type": "string",
                    "enum": valid_next,
                    "description": f"The character to act next. Previous speaker was {prev_name or '(none)'}. Do NOT pick the same character twice in a row unless the plot explicitly requires it.",
                },
                "directive": {
                    "type": "string",
                    "description": "Directive for the chosen character. Omit if the next character is the player.",
                },
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of suggestions. Omit if the next character is not the player.",
                },
                "enter_characters": {
                    "type": "array",
                    "items": {"type": "string", "enum": list_away} if list_away else {"type": "string"},
                    "description": f"List of characters to enter the scene. Valid options: {', '.join(list_away) if list_away else 'none'}.",
                },
                "exit_characters": {
                    "type": "array",
                    "items": {"type": "string", "enum": list_here},
                    "description": f"List of characters to exit the scene. Valid options: {', '.join(list_here)}.",
                },
                "switch_location": {
                    "type": "string",
                    "enum": valid_locs + [""],
                    "description": f"The location to switch to. Current: {loc.name}. Give empty string if no change.",
                },
                "edit_location": {
                    "type": "string",
                    "description": "True if characters or the plot alter the environment of the current location. "
                    "Provide a brief description of the modification (this is applied before switch_location). "
                    "Give an empty string if there is no change.",
                },
                "end_scene": {
                    "type": "boolean",
                    "description": "Set to true when the scene has reached its conclusion and should end. When true, next_scene MUST be set to a valid follow-up scene.",
                },
                "next_scene": {
                    "type": "string",
                    "enum": valid_scenes + [""],
                    "description": f"The follow-up scene to transition to. REQUIRED when end_scene is true. Provide empty string if the scene has not ended. Valid scenes: {', '.join(valid_scenes) if valid_scenes else 'none'}.",
                },
                "change_sprite": {
                    "type": "object",
                    "description": f"Optional: change the sprite for one or more on-screen characters. Keys are character names, values are sprite names.\nAvailable sprites:\n{sprite_info}",
                    "additionalProperties": {"type": "string"},
                },
            },
            required=[
                "next_character",
                "enter_characters",
                "exit_characters",
                "switch_location",
                "edit_location",
                "end_scene",
                "next_scene",
            ],
            strict=True,
        )

        recent_speakers_str = ", ".join(recent) if recent else "(none)"
        # Strong instruction to prevent back-to-back same speaker
        avoid_repeat = f"\nCRITICAL: The previous speaker was {prev_name}. You MUST NOT pick {prev_name} again. Choose a DIFFERENT character." if prev_name else ""

        # Build per-character briefing including prev-scene summaries.
        char_info_lines: list[str] = []
        for c in here_chars:
            parts = [
                f"{c.name} ({c.importance.name})",
                f"personality: {c.card_fields.get('personality', '')}",
                f"scenario: {c.card_fields.get('scenario', '')}",
            ]
            if c.prev_scene_summary:
                parts.append(f"orientation: {c.prev_scene_summary}")
            char_info_lines.append(", ".join(parts))

        away_info_lines: list[str] = []
        for c in away_chars:
            parts = [f"{c.name} ({c.importance.name})"]
            if c.prev_scene_summary:
                parts.append(f"orientation: {c.prev_scene_summary}")
            away_info_lines.append(", ".join(parts))

        plot_content = f"""Plot:
{scene.plot_as_tool_content()}

Characters currently here: {list_here}
Info:
{'\n'.join(char_info_lines)}

Available sprites per character:
{sprite_info}

Characters currently away: {list_away}
{'\n'.join(away_info_lines) if away_info_lines else '(no away characters with orientation)'}

Turn count: {turn_count}
Recent speakers: {recent_speakers_str}{avoid_repeat}

Note: ANONYMOUS characters are background extras with minimal persistence. They do not have detailed backstories or memory.
"""

        branch = ctx.branch()
        if branch.head is not None and branch.head.get("role") == "assistant":
            branch.user_message("Continue.", name="System")

        if history and isinstance(history, str):
            branch.user_message("Previously, in past scenes...", name="System")
            branch.assistant_message(str(history), name="System")

        branch.user_message("What is the current Plot?", name="System")
        branch.assistant_message(str(plot_content), name="System")

        logger.debug("Control handed to orchestrator")
        result = self.client.complete(
            role=GameRole.ORCHESTRATOR,
            system_prompt=self._system_prompt(scene.player, scene.narrator, scene),
            messages=branch.to_list(),
            tools=[next_round_tool],
            stream=True,
            print_stream=True,
        )
        print()

        if not result.tool_calls:
            logger.warning(
                f"Orchestrator returned no tool calls (content={result.content!r}). "
                f"Retries remaining: {_retries}"
            )
            if _retries > 0:
                return self.decide_next_turn(
                    scene, ctx, here_chars, away_chars, prev_char, loc, _retries - 1
                )
            raise RuntimeError("Orchestrator failed to produce a tool call after retries.")

        try:
            for tc in result.tool_calls:
                self.registry.call(tc["function"]["name"], tc["function"]["arguments"])
        except json.JSONDecodeError as exc:
            logger.warning(
                f"Orchestrator returned malformed JSON ({exc!r}). "
                f"Retries remaining: {_retries}"
            )
            if _retries > 0:
                return self.decide_next_turn(
                    scene, ctx, here_chars, away_chars, prev_char, loc,
                    history=history, turn_count=turn_count,
                    speaker_history=speaker_history, _retries=_retries - 1,
                )
            raise RuntimeError("Orchestrator failed to produce valid JSON after retries.")

        assert self._capture is not None
        decision = self._capture.to_decision(
            here_chars | {scene.narrator},
            away_chars,
            scene.location_pool,
            scene,
        )
        switch_name = decision.switch_location.name if decision.switch_location else "(none)"
        logger.info(
            f"Orchestrator decision: scene={scene.id}, loc={loc.name}, "
            f"next={decision.next_char.name}, "
            f"directive={decision.directive!r}, "
            f"suggestions={decision.suggestions!r}, "
            f"enter={[c.name for c in decision.entering_chars]}, "
            f"exit={[c.name for c in decision.exiting_chars]}, "
            f"switch_location={switch_name}, "
            f"edit_location={decision.edit_location!r}, "
            f"end_scene={decision.next_scene is not None}, "
            f"next_scene={decision.next_scene}, "
            f"change_sprite={decision.change_sprite}"
        )
        return decision


class _NextRoundCapture:
    """Internal helper that captures the arguments of the ``next_round`` tool."""

    def __init__(self) -> None:
        self._data: dict | None = None

    def hook(self, args: str) -> str:
        """Parse and store the JSON arguments.

        :param args: JSON-encoded argument string.
        :return: Empty string (tool result is discarded for the orchestrator).
        """
        self._data = json.loads(args)
        return ""

    def to_decision(
        self,
        present_chars: set[Character],
        away_chars: set[Character],
        loc_pool: set[Location],
        scene: Scene,
    ) -> TurnDecision:
        """Convert captured JSON into a typed :class:`TurnDecision`.

        Validates returned names against the valid pools since the tool schema
        no longer uses enums.

        :param present_chars: Characters currently in the scene (for next_char and exit).
        :param away_chars: Characters currently away from the scene (for enter).
        :param loc_pool: All locations the orchestrator could have selected.
        :param scene: Current scene (used for fallback values).
        :return: Resolved decision object.
        :raises RuntimeError: If the orchestrator failed to produce data or
            named an unknown character/location/scene.
        """
        if self._data is None:
            raise RuntimeError("Orchestrator did not produce a decision.")

        end_scene = self._data.get("end_scene", False)
        next_scene = self._data.get("next_scene", "")

        if end_scene or next_scene:
            if next_scene and next_scene not in scene.next_choices:
                raise RuntimeError(
                    f"Invalid next_scene '{next_scene}'. "
                    f"Valid options: {list(scene.next_choices.keys())}"
                )
            # If end_scene is true but next_scene is empty, auto-pick the first
            # valid scene as a fallback so the scene actually ends.
            if end_scene and not next_scene:
                valid_scenes = list(scene.next_choices.keys())
                if valid_scenes:
                    next_scene = valid_scenes[0]
                    logger.warning(
                        f"Orchestrator set end_scene=true but next_scene was empty. "
                        f"Auto-selected '{next_scene}'."
                    )
                else:
                    raise RuntimeError(
                        "end_scene is true but no valid next scenes are available."
                    )
            if next_scene:
                return TurnDecision(
                    next_char=scene.player,
                    directive="",
                    suggestions=[],
                    entering_chars=set(),
                    exiting_chars=set(),
                    switch_location=None,
                    next_scene=next_scene,
                )

        # Validate and resolve change_sprite
        change_sprite: dict[str, str] = {}
        raw_changes = self._data.get("change_sprite", {})
        if isinstance(raw_changes, dict):
            for char_name, sprite_name in raw_changes.items():
                char = next((c for c in present_chars if c.name == char_name), None)
                if char is None:
                    logger.warning(
                        f"Orchestrator tried to change sprite for unknown character '{char_name}'"
                    )
                    continue
                if sprite_name not in char.sprites:
                    logger.warning(
                        f"Orchestrator chose invalid sprite '{sprite_name}' for {char.name}. "
                        f"Valid: {char.sprites}"
                    )
                    continue
                change_sprite[char_name] = sprite_name

        next_name = self._data["next_character"]
        next_char = next((c for c in present_chars if c.name == next_name), None)
        if next_char is None:
            raise RuntimeError(
                f"Character '{next_name}' not found. "
                f"Valid options: {[c.name for c in present_chars]}"
            )

        def _find(names: list[str], pool: set[Character]) -> set[Character]:
            result: set[Character] = set()
            for n in names:
                char = next((c for c in pool if c.name == n), None)
                if char:
                    result.add(char)
                else:
                    logger.warning(
                        f"Orchestrator named unknown character '{n}' in "
                        f"enter/exit list. Valid options: {[c.name for c in pool]}"
                    )
            return result

        switch_name = self._data.get("switch_location", "")
        switch_loc = next((l for l in loc_pool if l.name == switch_name), None)
        if switch_name and switch_loc is None:
            raise RuntimeError(
                f"Location '{switch_name}' not found. "
                f"Valid options: {[l.name for l in loc_pool]}"
            )

        return TurnDecision(
            next_char=next_char,
            directive=self._data.get("directive", ""),
            suggestions=self._data.get("suggestions", []),
            entering_chars=_find(self._data.get("enter_characters", []), away_chars),
            exiting_chars=_find(self._data.get("exit_characters", []), present_chars),
            switch_location=switch_loc,
            edit_location=self._data.get("edit_location", ""),
            next_scene=None,
            change_sprite=change_sprite,
        )
