"""Multi-round conversation loop for a single scene."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.tools import ToolRegistry, tool
from ara.memory.chroma import ChromaStore
from ara.models import GameRole, Importance
from ara.world.character import Character
from ara.world.orchestrator import Orchestrator, TurnDecision
from ara.world.scene import Location, Scene
from ara.utils.debug import DebugConsole
from ara.utils.logger import get_logger

logger = get_logger(__name__)


def _character_system_prompt(char: Character, scene: Scene) -> str:
    """Build a system prompt for an NPC turn.

    :param char: The character who will speak.
    :param scene: Current scene (provides language and tone).
    :return: Formatted system prompt.
    """
    return f"""IMPORTANT: Reply in {scene.language} only!
# Role
 - You are {char.name}.
 - Write how you think {char.name} would reply based on {char.name}'s previous messages.
 - Never write as the other character(s) or as the Narrator.

## Format
 - Do not prefix your response with your name.
 - Use newlines to separate speech from actions.
"""


def _narrator_system_prompt(
    player: Character,
    narrator: Character,
    scene: Scene,
) -> str:
    """Build a system prompt for a narrator turn.

    :param player: Player-controlled character.
    :param narrator: Narrator character.
    :param scene: Current scene.
    :return: Formatted system prompt.
    """
    return f"""IMPORTANT: Reply in {scene.language} only!
# Role: Visual Novel Narrator
## Core Purpose
You are the {narrator.name}, the Narrator of the visual novel.
The player is {player.name}.

## Narrative Rules
1. **Content Scope**:
   - Be EXTREMELY concise. Write exactly ONE sentence. Maximum 30 words.
   - Do NOT write long paragraphs of atmospheric description.
   - Express unspoken character thoughts (only for {player.name}).
   - Handle scene transitions when directed.

2. **Style Guidelines**:
   - Do not prefix your response with your name.
   - Match the plot zeitgeist: {scene.zeitgeist}.
   - Match the scene tone: {scene.tone}.
   - Never speak for characters.

## Prohibitions
 - Never advance plot through character dialogue.
 - Never describe active character actions (reserved for character agents).
 - NEVER write more than one sentence.
"""


@dataclass
class EngineStepResult:
    """Result of a single :meth:`Engine.step` call.

    :ivar scene_ended: ``True`` when the orchestrator has chosen a next scene.
    :ivar next_scene: Identifier of the next scene (only when *scene_ended*).
    :ivar needs_player_input: ``True`` when the turn belongs to the player and
        the engine is waiting for a call to :meth:`submit_player_input`.
    :ivar suggestions: Orchestrator suggestions for the player (only when
        *needs_player_input*).
    :ivar enter: Names of characters that entered this turn.
    :ivar exit: Names of characters that exited this turn.
    :ivar sprite_changes: Mapping of character names → new sprite names applied
        this turn.
    """

    scene_ended: bool = False
    next_scene: str | None = None
    needs_player_input: bool = False
    suggestions: list[str] = field(default_factory=list)
    speaker: str | None = None
    enter: list[str] = field(default_factory=list)
    exit: list[str] = field(default_factory=list)
    sprite_changes: dict[str, str] = field(default_factory=dict)


class Engine:
    """Drives the multi-turn conversation within a single scene.

    The engine repeatedly asks the :class:`Orchestrator` for a
    :class:`TurnDecision`, then dispatches to the narrator, player, or an NPC
    accordingly.  Character tool calls (scratch updates, memory recall) are
    executed inline and their results are appended to the conversation context.

    Usage (state-machine style)::

        engine = Engine(client)
        engine.start(scene)
        while not engine.finished:
            result = engine.step()
            if result.needs_player_input:
                text = input(f"{engine.scene.player.name}> ")
                engine.submit_player_input(text)
        next_scene = engine.next_scene

    The old blocking :meth:`run` convenience wrapper is still available.
    """

    def __init__(self, client: LLMClient, db: ChromaStore | None = None) -> None:
        self.client = client
        self.db = db
        self.orchestrator = Orchestrator(client)
        self._debug: DebugConsole | None = None
        self._last_decision: TurnDecision | None = None
        self._here_chars: set[Character] = set()
        self._away_chars: set[Character] = set()
        self._loc: Location | None = None
        self._scene: Scene | None = None
        self._ctx: ConversationContext | None = None
        self._prev_char: Character | None = None
        self._next_scene: str | None = None
        self._running = False
        self._needs_player_input = False
        self._directives_log: dict[Character, str] = {}
        self._turn_count: int = 0
        self._speaker_history: list[str] = []

    # ------------------------------------------------------------------ #
    # Public state for inspection
    # ------------------------------------------------------------------ #

    @property
    def finished(self) -> bool:
        """``True`` when the orchestrator has signalled a scene transition."""
        return not self._running

    @property
    def needs_player_input(self) -> bool:
        """``True`` when the engine is waiting for :meth:`submit_player_input`."""
        return self._needs_player_input

    @property
    def next_scene(self) -> str | None:
        """Identifier of the next scene, or ``None`` if the story ends here."""
        return self._next_scene

    @property
    def scene(self) -> Scene | None:
        """The scene currently being played."""
        return self._scene

    @property
    def ctx(self) -> ConversationContext | None:
        """The active conversation context."""
        return self._ctx

    @property
    def here_chars(self) -> set[Character]:
        """Characters currently present."""
        return self._here_chars

    @property
    def away_chars(self) -> set[Character]:
        """Characters currently away from the scene."""
        return self._away_chars

    @property
    def loc(self) -> Location | None:
        """Current location."""
        return self._loc

    @property
    def last_decision(self) -> TurnDecision | None:
        """The most recent orchestrator decision."""
        return self._last_decision

    @property
    def directives_log(self) -> dict[Character, str]:
        """Mapping of character → last directive received this scene."""
        return self._directives_log

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def set_debug_console(self, debug_console: DebugConsole | None) -> None:
        """Attach or detach the debug console (used by :meth:`run`)."""
        self._debug = debug_console

    def start(self, scene: Scene) -> None:
        """Prepare the engine to drive *scene*.

        Must be called before the first :meth:`step`.
        """
        self._scene = scene
        self._here_chars = set(scene.starting_characters)
        self._away_chars = set(scene.character_pool) - self._here_chars
        self._loc = scene.starting_location
        self._prev_char = None
        self._last_decision = None
        self._next_scene = None
        self._running = True
        self._needs_player_input = False
        self._directives_log = {}
        self._turn_count = 0
        self._speaker_history = []
        self._ctx = ConversationContext(*[c.name for c in scene.character_pool])
        self._ctx.enter_entities(*[c.name for c in self._here_chars])

    def step(self) -> EngineStepResult:
        """Advance the conversation by one turn.

        When the orchestrator selects the player, the engine enters a waiting
        state and returns :attr:`EngineStepResult.needs_player_input` set to
        ``True``.  The caller must then call :meth:`submit_player_input`
        before the next :meth:`step`.

        :return: Description of what happened this tick.
        :raises RuntimeError: If the engine has not been started, is already
            finished, or is still waiting for player input.
        """
        if not self._running or self._scene is None or self._ctx is None:
            raise RuntimeError("Engine not started or already finished")
        if self._needs_player_input:
            raise RuntimeError(
                "Engine is waiting for player input. Call submit_player_input() first."
            )

        scene = self._scene
        ctx = self._ctx
        here_chars = self._here_chars
        away_chars = self._away_chars
        loc = self._loc

        logger.debug(
            f"Round start: here={[c.name for c in here_chars]}, "
            f"away={[c.name for c in away_chars]}, loc={loc.name if loc else 'None'}"
        )

        # Query story history for relevant past scenes
        history_text = ""
        if self.db is not None:
            try:
                results = self.db.query(
                    "story_history",
                    query_texts=[scene.plot_story],
                    n_results=3,
                )
                docs = results.get("documents", [[]]) or []
                summaries: list[str] = []
                for group in docs:
                    summaries.extend(group or [])
                if summaries:
                    history_text = "\n\n".join(summaries)
            except Exception as exc:
                logger.debug(f"Story history query failed: {exc}")

        decision = self.orchestrator.decide_next_turn(
            scene=scene,
            ctx=ctx,
            here_chars=here_chars,
            away_chars=away_chars,
            prev_char=self._prev_char,
            loc=loc or scene.starting_location,
            history=history_text,
            turn_count=self._turn_count,
            speaker_history=self._speaker_history,
        )

        # Apply sprite changes from the orchestrator decision.
        sprite_changes: dict[str, str] = {}
        for char_name, sprite_name in decision.change_sprite.items():
            char = next((c for c in here_chars if c.name == char_name), None)
            if char is not None:
                char.current_sprite = sprite_name
                sprite_changes[char_name] = sprite_name

        if decision.next_scene is not None:
            self._next_scene = decision.next_scene
            self._running = False
            return EngineStepResult(
                scene_ended=True, next_scene=decision.next_scene,
                speaker=decision.next_char.name,
                enter=[c.name for c in decision.entering_chars],
                exit=[c.name for c in decision.exiting_chars],
                sprite_changes=sprite_changes,
            )

        self._last_decision = decision
        self._prev_char = decision.next_char

        if decision.next_char != scene.player and decision.directive:
            self._directives_log[decision.next_char] = decision.directive

        if decision.switch_location is not None and decision.switch_location != loc:
            self._loc = decision.switch_location
            loc = self._loc

        if decision.edit_location and loc is not None:
            loc.desc += f"\n\n[Update]: {decision.edit_location}"
            logger.info(f"Location '{loc.name}' updated: {decision.edit_location}")

        enter_names = [c.name for c in decision.entering_chars]
        exit_names = [c.name for c in decision.exiting_chars]

        if decision.entering_chars:
            ctx.enter_entities(*enter_names)
            here_chars |= decision.entering_chars
            away_chars -= decision.entering_chars

        if decision.next_char == scene.player:
            self._needs_player_input = True
            return EngineStepResult(
                needs_player_input=True, suggestions=decision.suggestions,
                speaker=decision.next_char.name,
                enter=enter_names,
                exit=exit_names,
                sprite_changes=sprite_changes,
            )

        if decision.next_char == scene.narrator:
            logger.info(f"{decision.next_char.name} [Narrator] is speaking now.")
            self._narrator_turn(scene, ctx, decision, here_chars, away_chars, loc)
        else:
            logger.info(f"{decision.next_char.name} is speaking now.")
            self._character_turn(scene, ctx, decision, loc)

        if decision.exiting_chars:
            ctx.exit_entities(*exit_names)
            here_chars -= decision.exiting_chars
            away_chars |= decision.exiting_chars

        self._turn_count += 1
        self._speaker_history.append(decision.next_char.name)

        return EngineStepResult(
            speaker=decision.next_char.name,
            enter=enter_names,
            exit=exit_names,
            sprite_changes=sprite_changes,
        )

    def submit_player_input(self, text: str) -> None:
        """Provide the player's message and resume the engine.

        Must only be called after :meth:`step` returned
        ``needs_player_input=True``.

        :param text: Raw player input.
        :raises RuntimeError: If the engine is not waiting for input.
        """
        if not self._needs_player_input or self._scene is None or self._ctx is None:
            raise RuntimeError("Engine is not waiting for player input")
        self._ctx.user_message(text, name=self._scene.player.name)
        if text.strip() and self._scene.player.importance >= Importance.IMPORTANT:
            self._scene.player.memory.add_conversation([text.strip()])

        self._needs_player_input = False

    def generate_player_input(self, suggestion: str) -> str:
        """Generate natural player dialogue based on a suggestion.

        Uses a one-off LLM call to expand a suggestion/intent into the
        player's actual spoken words.

        :param suggestion: The suggestion text (e.g. "Ask about the Übermensch").
        :return: Generated player dialogue.
        :raises RuntimeError: If the engine is not waiting for input.
        """
        if not self._needs_player_input or self._scene is None or self._ctx is None:
            raise RuntimeError("Engine is not waiting for player input")
        scene = self._scene
        player = scene.player
        prompt = (
            f"You are {player.name} in a visual novel scene.\n"
            f"Zeitgeist: {scene.zeitgeist}\n"
            f"Tone: {scene.tone}\n"
            f"Language: {scene.language}\n\n"
            f"Write a brief, natural spoken response (1–3 sentences) "
            f"that matches this intent: {suggestion}\n\n"
            f"Respond with ONLY the dialogue. No quotes, no narration, "
            f"no stage directions."
        )
        result = self.client.complete(
            role=GameRole.CHARACTER,
            system_prompt=prompt,
            messages=[{"role": "user", "content": "Write the response."}],
            stream=False,
        )
        return result.content.strip()

    def run(
        self,
        scene: Scene,
        get_user_input: Callable[[str, list[str]], str] | None = None,
        debug_console: DebugConsole | None = None,
    ) -> str | None:
        """Run *scene* to completion (convenience wrapper).

        This is equivalent to calling :meth:`start` then looping :meth:`step`
        until :attr:`finished` becomes ``True``.  Player input is obtained
        through the optional *get_user_input* callback.

        :param scene: The scene to play through.
        :param get_user_input: Callback that receives a prompt string and a
            list of suggestions, and returns the player's typed input.  If
            omitted and a player turn occurs, a :exc:`RuntimeError` is raised.
        :param debug_console: Optional debug console (forwarded to
            :meth:`set_debug_console`).
        :return: The identifier of the next scene chosen by the orchestrator,
            or ``None`` if the story ends here.
        """
        self.set_debug_console(debug_console)
        self.start(scene)
        while not self.finished:
            if self._debug and self._debug.auto_pause:
                self._debug.pause(
                    scene=self._scene,
                    ctx=self._ctx,
                    here_chars=self._here_chars,
                    away_chars=self._away_chars,
                    loc=self._loc or self._scene.starting_location,
                    decision=self._last_decision,
                )

            result = self.step()

            if result.needs_player_input:
                if get_user_input is None:
                    raise RuntimeError(
                        "Engine needs player input but no get_user_input was provided. "
                        "Use the step()/submit_player_input() API instead."
                    )
                while True:
                    user_text = get_user_input(
                        f"{self._scene.player.name}> ", result.suggestions
                    )
                    stripped = user_text.strip()
                    if stripped.startswith(("/", ":")):
                        if self._debug is not None:
                            self._debug.pause(
                                scene=self._scene,
                                ctx=self._ctx,
                                here_chars=self._here_chars,
                                away_chars=self._away_chars,
                                loc=self._loc or self._scene.starting_location,
                                decision=self._last_decision,
                                noshell=stripped[1:],
                            )
                        else:
                            print("Debug console not available. Start with --debug-console.")
                        continue
                    break
                self.submit_player_input(user_text)

        return self.next_scene

    def _narrator_turn(
        self,
        scene: Scene,
        ctx: ConversationContext,
        decision: TurnDecision,
        here_chars: set[Character],
        away_chars: set[Character],
        loc: Location,
    ) -> None:
        """Execute a narrator turn.

        The narrator receives the location scratch as injected context and
        writes environmental or atmospheric text.
        """
        branch = ctx.branch()
        branch.concat_context(loc.scratch_context())
        branch.user_message(
            f"Characters present: {[c.name for c in here_chars]}\n"
            f"Characters away: {[c.name for c in away_chars]}\n"
            f"Directive: {decision.directive or 'None'}",
            name="System",
        )

        result = self.client.complete(
            role=GameRole.NARRATOR,
            system_prompt=_narrator_system_prompt(
                scene.player, scene.narrator, scene
            ),
            messages=branch.to_list(),
            stream=True,
            print_stream=True,
        )

        print()

        ctx.assistant_message(
            result.content,
            tool_calls=[],
            name=scene.narrator.name,
            reasoning_content=result.reasoning_content,
        )
        if result.content.strip() and scene.narrator.importance >= Importance.IMPORTANT:
            scene.narrator.memory.add_conversation([result.content.strip()])

    def _player_turn(
        self,
        scene: Scene,
        ctx: ConversationContext,
        decision: TurnDecision,
        get_user_input: Callable[[str, list[str]], str],
    ) -> None:
        """Execute a player turn.

        Displays the orchestrator's suggestions and reads a line from stdin.
        If the user types ``/command`` or ``:command``, the text is routed to
        the debug console instead of being treated as in-character input.
        """
        if decision.suggestions:
            print("\n".join(decision.suggestions))
        prompt = f"{scene.player.name}> "

        while True:
            user_text = get_user_input(prompt, decision.suggestions)
            stripped = user_text.strip()
            if stripped.startswith(("/", ":")):
                if self._debug is not None:
                    self._debug.pause(
                        scene=scene,
                        ctx=ctx,
                        here_chars=self._here_chars,
                        away_chars=self._away_chars,
                        loc=self._loc or scene.starting_location,
                        decision=decision,
                        noshell=stripped[1:],
                    )
                else:
                    print("Debug console not available. Start with --debug-console.")
                continue
            break

        ctx.user_message(user_text, name=scene.player.name)

    def _character_turn(
        self,
        scene: Scene,
        ctx: ConversationContext,
        decision: TurnDecision,
        loc: Location,
    ) -> None:
        """Execute an NPC turn.

        The character receives their ``whoami`` as injected context.
        Important characters also receive the location scratch.
        Characters may use tools (recall, think, write_scratch) before speaking.
        """
        char = decision.next_char

        # ---- character tools ------------------------------------------------
        recall_tool = tool(
            name="recall",
            description=(
                "Search YOUR OWN memory for relevant past conversations or events. "
                "This only returns memories from your personal perspective — you cannot "
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

        think_tool = tool(
            name="think",
            description="Record a private thought or observation. This is not spoken aloud.",
            properties={
                "thought": {
                    "type": "string",
                    "description": "Your internal thought. Other characters cannot hear this.",
                }
            },
            required=["thought"],
            strict=True,
        )

        write_scratch_tool = tool(
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

        registry = ToolRegistry()
        has_tools = char.importance >= Importance.IMPORTANT

        def _recall_handler(args: str) -> str:
            data = json.loads(args)
            query = data.get("query", "")
            memories = char.memory.recall([query], depth="medium")
            if memories:
                return "\n".join(f"- {m}" for m in memories)
            return "You don't recall anything relevant."

        def _think_handler(args: str) -> str:
            data = json.loads(args)
            thought = data.get("thought", "")
            if thought:
                if char.scratch.text == "Nothing yet!":
                    char.scratch.text = f"[Thought]: {thought}"
                else:
                    char.scratch.text += f"\n[Thought]: {thought}"
            return "Thought recorded."

        def _write_scratch_handler(args: str) -> str:
            data = json.loads(args)
            note = data.get("note", "")
            if note:
                if char.scratch.text == "Nothing yet!":
                    char.scratch.text = f"[Note]: {note}"
                else:
                    char.scratch.text += f"\n[Note]: {note}"
            return "Note saved."

        character_tools: list[dict] = []
        if has_tools:
            registry.register("recall", _recall_handler)
            registry.register("think", _think_handler)
            registry.register("write_scratch", _write_scratch_handler)
            character_tools = [recall_tool, think_tool, write_scratch_tool]
        # -------------------------------------------------------------------- #

        def _build_branch() -> ConversationContext:
            b = ctx.branch()
            b.concat_context(char.whoami)
            b.filter_to(char.name)
            if char.importance >= Importance.IMPORTANT:
                b.concat_context(loc.scratch_context())
            b.concat_context(char.scene_summary_context)
            b.user_message(
                f"Current directive: {decision.directive or 'None'}",
                name="System",
            )
            return b

        branch = _build_branch()
        result = self.client.complete(
            role=GameRole.CHARACTER,
            system_prompt=_character_system_prompt(char, scene),
            messages=branch.to_list(),
            tools=character_tools or None,
            tool_choice="auto" if has_tools else None,
            stream=True,
            print_stream=True,
        )

        # Handle tool calls: execute, append to base context, re-call LLM.
        # The model may chain multiple tool calls (e.g. recall → recall → think)
        # before finally producing spoken content.
        while result.tool_calls:
            ctx.assistant_message(
                result.content,
                tool_calls=result.tool_calls,
                name=char.name,
                reasoning_content=result.reasoning_content,
            )
            for tc in result.tool_calls:
                result_text = registry.call(
                    tc["function"]["name"], tc["function"]["arguments"]
                )
                ctx.tool_message(result_text, tool_call_id=tc["id"])

            branch2 = _build_branch()
            result = self.client.complete(
                role=GameRole.CHARACTER,
                system_prompt=_character_system_prompt(char, scene),
                messages=branch2.to_list(),
                tools=character_tools,
                tool_choice="auto" if has_tools else None,
                stream=True,
                print_stream=True,
            )

        print()

        ctx.assistant_message(
            result.content,
            tool_calls=[],
            name=char.name,
            reasoning_content=result.reasoning_content,
        )
        if result.content.strip() and char.importance >= Importance.IMPORTANT:
            char.memory.add_conversation([result.content.strip()])
