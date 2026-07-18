"""Multi-round conversation loop for a single scene."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.tools import ToolRegistry, tool
from ara.memory.chroma import ChromaStore
from ara.llm.models import GameRole
from ara.world.character import Importance
from ara.world.character import Character, create_anonymous_character
from ara.world.orchestrator import Orchestrator, TurnDecision
from ara.world.scene import Location, Scene
from ara.world.system_page import SystemPage

from ara.utils.debug import DebugConsole
from ara.utils.logger import get_logger

logger = get_logger(__name__)


from ara.prompts.character import character_system_prompt as _character_system_prompt
from ara.prompts.narrator import narrator_system_prompt as _narrator_system_prompt


def _hidden_not_visible(observer: Character, here_chars: set[Character]) -> set[str]:
    """Canonical names of hidden characters that *observer* cannot perceive."""
    observer_canonical = observer.canonical_name
    return {
        c.canonical_name for c in here_chars
        if c.hidden
        and observer_canonical != c.canonical_name
        and observer_canonical not in c.visible_to
    }


def _parse_inner_response(
    content: str, mode: str = "outer"
) -> tuple[str, str | None, str | None]:
    """Try to parse a response with public outer text and private inner thought.

    Expected JSON form for ``outer_and_inner``:
    {"outer": "...", "inner": "...", "explain": "..."}.

    For ``inner_only``, the whole response is treated as the private thought
    unless it is valid JSON with an ``inner`` field.

    On failure in ``outer`` mode, returns the raw content as the outer text.

    :return: (outer, inner, explain)
    """
    content = content.strip()
    if mode == "inner_only":
        if not content.startswith("{"):
            return "", content, None
        try:
            data = json.loads(content)
            if isinstance(data, dict) and isinstance(data.get("inner"), str):
                return "", data["inner"], data.get("explain")
        except json.JSONDecodeError:
            pass
        return "", content, None

    if not content.startswith("{"):
        return content, None, None
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            return content, None, None
        outer = data.get("outer", "")
        if not isinstance(outer, str):
            outer = ""
        inner = data.get("inner")
        if not isinstance(inner, str):
            inner = None
        explain = data.get("explain")
        if not isinstance(explain, str):
            explain = None
        return outer, inner, explain
    except json.JSONDecodeError:
        return content, None, None


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
    :ivar spawn: Names of anonymous characters spawned this turn.
    :ivar sprite_changes: Mapping of character names → new sprite names applied
        this turn.
    :ivar switch_background: Background stem activated for the current location,
        or empty string if unchanged.
    :ivar system_changes: Updates to the player system page applied this turn.
    :ivar output: Text produced by this turn (spoken dialogue or narration).
        Empty for non-speech ticks such as scene transitions.
    :ivar inner: Private inner thought produced by this turn, if any.
    """

    scene_ended: bool = False
    next_scene: str | None = None
    needs_player_input: bool = False
    suggestions: list[str] = field(default_factory=list)
    speaker: str | None = None
    speaker_title: str = ""
    enter: list[str] = field(default_factory=list)
    exit: list[str] = field(default_factory=list)
    spawn: list[str] = field(default_factory=list)
    sprite_changes: dict[str, str] = field(default_factory=dict)
    switch_background: str = ""
    system_changes: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    inner: str = ""


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
        """Create an engine.

        :param client: LLM client used for all model calls.
        :param db: Optional ChromaDB store for story-history retrieval.
        """
        self.client = client
        self.db = db
        self.orchestrator = Orchestrator(client, db=db)
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
        self._world_time: str = ""
        self._player_status: dict[str, Any] = {}
        self._mechanical_changelog: list[dict[str, Any]] = []
        self._story_state: dict[str, Any] = {}
        self._turn_count: int = 0
        self._speaker_history: list[str] = []
        self._canonical_events: list[dict] = []
        self._canonical_index: int = 0
        self._free_status: dict[str, Any] = {}
        self._pending_attempts: list[dict[str, Any]] = []

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

    @property
    def world_time(self) -> str:
        """Current world time (e.g. 'morning', 'night')."""
        return self._world_time

    @property
    def player_status(self) -> dict[str, Any]:
        """Player system-page state (bars, inventory, skills, etc.)."""
        return self._player_status

    @property
    def mechanical_changelog(self) -> list[dict[str, Any]]:
        """Mechanical state changes applied during the current scene."""
        return self._mechanical_changelog

    @property
    def free_status(self) -> dict[str, Any]:
        """Miscellaneous world status not tied to the player or a location."""
        return self._free_status

    # ------------------------------------------------------------------ #
    # Status-page helpers
    # ------------------------------------------------------------------ #

    def _resolve_inventory_item(self, item: Any) -> Any:
        """Fill missing name/description/metadata from a plot item template."""
        if not isinstance(item, dict) or "id" not in item or self._scene is None:
            return item
        template = self._scene.items.get(item["id"])
        if template is None:
            return item
        merged = dict(item)
        if not merged.get("name"):
            merged["name"] = template.name
        if not merged.get("description"):
            merged["description"] = template.description
        if not merged.get("icon"):
            if template.icon:
                merged["icon"] = template.icon
        if template.metadata:
            merged_metadata = dict(template.metadata)
            merged_metadata.update(merged.get("metadata", {}))
            merged["metadata"] = merged_metadata
        return merged

    def _apply_page_update(
        self, page_dict: dict[str, Any], changes: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge a status-page update into *page_dict* and return the new dict."""
        page = SystemPage.from_dict(page_dict)
        if "sections" in changes or "title" in changes:
            new_page = SystemPage.from_dict(changes)
            if new_page.title:
                page.title = new_page.title
            existing: dict[str, dict[str, Any]] = {}
            for s in page.sections:
                key = s.get("label")
                if key:
                    key = f"{s.get('type')}:{key}"
                else:
                    key = s.get("type")
                existing[key] = s
            for section in new_page.sections:
                stype = section.get("type")
                if stype == "inventory":
                    section = dict(section)
                    section["items"] = [
                        self._resolve_inventory_item(it)
                        for it in section.get("items", [])
                    ]
                key = section.get("label")
                if key:
                    key = f"{stype}:{key}"
                else:
                    key = stype
                prev = existing.get(key)
                if prev and prev.get("items") and section.get("items"):
                    # Merge items by their label, replacing matches
                    merged = []
                    seen: set[str] = set()

                    def _item_label(item: Any) -> str:
                        if isinstance(item, dict):
                            return str(item.get("label", ""))
                        return str(item)

                    for item in section.get("items", []):
                        merged.append(item)
                        seen.add(_item_label(item))
                    for item in prev.get("items", []):
                        if _item_label(item) not in seen:
                            merged.append(item)
                    existing[key] = dict(section, items=merged)
                else:
                    existing[key] = section
            page.sections = list(existing.values())
        else:
            legacy = page.to_legacy()
            legacy.update(changes)
            page = SystemPage.from_legacy(legacy)
        return page.to_dict()

    def _resolve_status_target(self, target: str) -> tuple[dict[str, Any], str]:
        """Return the status-page dict and label for *target*.

        *target* may be a canonical ID or a display name.
        """
        scene = self._scene
        if target == "player":
            return self._player_status, "player"
        if target == "free":
            return self._free_status, "free"
        loc_target = scene.location_by_name(target) if scene else None
        if loc_target is not None:
            return loc_target.status, f"location {loc_target.canonical_name}"
        char_target = scene.character_by_name(target) if scene else None
        if char_target is not None:
            return char_target.status, f"character {char_target.canonical_name}"
        logger.warning(f"update_status_page target '{target}' not found; defaulting to player")
        return self._player_status, "player"

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
        self._world_time = scene.time
        self._player_status = {}
        self._mechanical_changelog = []
        self._free_status = {}
        self._pending_attempts = []
        self._story_state = {}
        self._turn_count = 0
        self._speaker_history = []
        self._ctx = ConversationContext(*[c.canonical_name for c in scene.character_pool])
        self._ctx.enter_entities(*[c.canonical_name for c in self._here_chars])
        self._canonical_events = scene.canonical_events
        self._canonical_index = 0
        self._canonical_pending_choices: list[dict] | None = None

    def _canonical_step(self) -> EngineStepResult:
        """Execute the next scripted event from the canonical queue."""
        assert self._scene is not None and self._ctx is not None
        scene = self._scene
        ctx = self._ctx
        here_chars = self._here_chars
        away_chars = self._away_chars
        loc = self._loc

        if self._canonical_index >= len(self._canonical_events):
            # No more events - auto-end the scene if possible
            if self._next_scene is None and scene.next_choices:
                self._next_scene = next(iter(scene.next_choices))
            self._running = False
            return EngineStepResult(
                scene_ended=True,
                next_scene=self._next_scene,
                speaker=None,
                switch_background="",
                system_changes={},
            )

        ev = self._canonical_events[self._canonical_index]
        self._canonical_index += 1

        event_type = ev.get("event", "")
        speaker_name = ev.get("speaker", "")
        speaker = scene.character_by_name(speaker_name) if speaker_name else None
        output = ev.get("output", "")
        enter_names = list(ev.get("enter", []))
        exit_names = list(ev.get("exit", []))
        sprite_changes = dict(ev.get("sprite_changes", {}))
        switch_location_name = ev.get("switch_location")
        switch_background_name = ev.get("switch_background", "")
        edit_location = ev.get("edit_location", "")

        # Resolve characters by canonical or display name for backwards compatibility.
        entering_chars = {scene.character_by_name(n) for n in enter_names}
        entering_chars.discard(None)
        exiting_chars = {scene.character_by_name(n) for n in exit_names}
        exiting_chars.discard(None)

        # Apply sprite changes
        for char_name, sprite_name in sprite_changes.items():
            char = scene.character_by_name(char_name)
            if char:
                char.current_sprite = sprite_name

        # Apply location switch
        if switch_location_name:
            new_loc = scene.location_by_name(switch_location_name)
            if new_loc is not None:
                self._loc = new_loc
                loc = self._loc

        # Apply background switch
        if switch_background_name and loc is not None:
            if switch_background_name in loc.backgrounds:
                loc.current_background = switch_background_name
            else:
                logger.warning(
                    f"Canonical event: invalid background '{switch_background_name}' "
                    f"for location '{loc.name}'. Valid: {loc.backgrounds}"
                )

        # Apply location edit
        if edit_location and loc is not None:
            loc.desc += f"\n\n[Update]: {edit_location}"

        enter_canonical = [c.canonical_name for c in entering_chars]
        exit_canonical = [c.canonical_name for c in exiting_chars]

        # Apply enters BEFORE turn
        if entering_chars:
            ctx.enter_entities(*enter_canonical)
            here_chars |= entering_chars
            away_chars -= entering_chars

        # Process the turn
        if event_type == "turn":
            if speaker is None:
                raise RuntimeError(f"Canonical turn {self._canonical_index}: unknown speaker '{speaker_name}'")

            if speaker == scene.player:
                ctx.user_message(
                    output,
                    name=scene.player.name,
                    hidden=scene.player.hidden,
                    visible_to=set(scene.player.visible_to) if scene.player.hidden else None,
                    canonical_name=scene.player.canonical_name,
                )
                if output.strip() and scene.player.importance >= Importance.IMPORTANT:
                    scene.player.memory.add_conversation([output.strip()])
            else:
                ctx.assistant_message(
                    output,
                    tool_calls=[],
                    name=speaker.name,
                    hidden=speaker.hidden,
                    visible_to=set(speaker.visible_to) if speaker.hidden else None,
                    canonical_name=speaker.canonical_name,
                )
                if output.strip() and speaker.importance >= Importance.IMPORTANT:
                    speaker.memory.add_conversation([output.strip()])

            self._turn_count += 1
            self._speaker_history.append(speaker.canonical_name)
            self._prev_char = speaker

        elif event_type == "scene_ended":
            choices = ev.get("choices")
            if choices:
                self._canonical_pending_choices = choices
                self._needs_player_input = True
                if exiting_chars:
                    ctx.exit_entities(*exit_canonical)
                    here_chars -= exiting_chars
                    away_chars |= exiting_chars
                return EngineStepResult(
                    needs_player_input=True,
                    suggestions=[c["hint"] for c in choices],
                    speaker=speaker.canonical_name if speaker else speaker_name,
                    enter=enter_canonical,
                    exit=exit_canonical,
                    sprite_changes=sprite_changes,
                    switch_background="",
                    system_changes={},
                    output="",
                )
            next_scene = ev.get("next_scene")
            if next_scene:
                self._next_scene = next_scene
            self._running = False

        elif event_type == "story_complete":
            self._running = False

        # Apply exits AFTER turn
        if exiting_chars:
            ctx.exit_entities(*exit_canonical)
            here_chars -= exiting_chars
            away_chars |= exiting_chars

        if event_type == "scene_ended":
            return EngineStepResult(
                scene_ended=True,
                next_scene=self._next_scene,
                speaker=speaker.canonical_name if speaker else speaker_name,
                enter=enter_canonical,
                exit=exit_canonical,
                sprite_changes=sprite_changes,
                switch_background="",
                system_changes={},
                output="",
            )
        elif event_type == "story_complete":
            return EngineStepResult(
                scene_ended=True,
                speaker=speaker.canonical_name if speaker else speaker_name,
                enter=enter_canonical,
                exit=exit_canonical,
                sprite_changes=sprite_changes,
                switch_background="",
                system_changes={},
                output="",
            )
        else:
            return EngineStepResult(
                speaker=speaker.canonical_name if speaker else speaker_name,
                enter=enter_canonical,
                exit=exit_canonical,
                sprite_changes=sprite_changes,
                switch_background="",
                system_changes={},
                output=output,
            )

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
        if self._canonical_events:
            return self._canonical_step()

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

        attempts_for_orchestrator = list(self._pending_attempts)
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
            story_state=self._story_state,
            attempts_for_orchestrator=attempts_for_orchestrator,
            player_status=self._player_status,
            free_status=self._free_status,
        )
        self._pending_attempts.clear()
        return self._apply_decision(decision)


    def submit_attempt(self, attempt: str | dict[str, Any], source: str | None = None) -> None:
        """Store an action attempt for the orchestrator.

        This does NOT end the player turn. The attempt will be shown to the
        orchestrator on the next turn decision and then cleared.

        :param attempt: Either a plain text action or a dict with fields such as
            ``action``, ``intent``, ``target``, and ``secrecy``.
        :param source: Name of the character submitting the attempt (defaults to
            the player when called during a player turn).
        """
        if isinstance(attempt, str):
            entry: dict[str, Any] = {"action": attempt}
        else:
            entry = dict(attempt)
        entry.setdefault("source", source or (self._scene.player.canonical_name if self._scene else "Unknown"))
        self._pending_attempts.append(entry)

    def submit_player_input(
        self,
        text: str,
        attempt: str | dict[str, Any] | None = None,
    ) -> None:
        """Provide the player's message and resume the engine.

        Must only be called after :meth:`step` returned
        ``needs_player_input=True``.

        In canonical mode with end-of-scene choices, *text* is matched against
        the predetermined choice messages to set :attr:`next_scene`.

        :param text: Raw player input.
        :param attempt: Optional action attempt to record alongside the reply.
        :raises RuntimeError: If the engine is not waiting for input.
        """
        if not self._needs_player_input or self._scene is None or self._ctx is None:
            raise RuntimeError("Engine is not waiting for player input")

        if attempt:
            self.submit_attempt(attempt, source=self._scene.player.canonical_name)

        if self._canonical_pending_choices is not None:
            for choice in self._canonical_pending_choices:
                if text == choice["text"] or text == choice.get("hint", ""):
                    self._next_scene = choice["next_scene"]
                    break
            else:
                valid = [c["text"] for c in self._canonical_pending_choices]
                raise RuntimeError(
                    f"Invalid canonical choice: '{text}'. Valid: {valid}"
                )
            self._canonical_pending_choices = None

        self._ctx.user_message(
            text,
            name=self._scene.player.name,
            hidden=self._scene.player.hidden,
            visible_to=set(self._scene.player.visible_to) if self._scene.player.hidden else None,
            canonical_name=self._scene.player.canonical_name,
        )
        if text.strip() and self._scene.player.importance >= Importance.IMPORTANT:
            self._scene.player.memory.add_conversation([text.strip()])

        self._speaker_history.append(self._scene.player.canonical_name)
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
            name=player.name,
        )
        return result.content.strip()

    def _narrator_turn(
        self,
        scene: Scene,
        ctx: ConversationContext,
        decision: TurnDecision,
        here_chars: set[Character],
        away_chars: set[Character],
        loc: Location,
    ) -> str:
        """Execute a narrator turn.

        The narrator receives the location scratch as injected context and
        writes environmental or atmospheric text.

        :return: The narrator's spoken text.
        """
        hidden_from_narrator = _hidden_not_visible(scene.narrator, here_chars)
        visible_here = [c for c in here_chars if c.canonical_name not in hidden_from_narrator]
        branch = ctx.branch()
        # The narrator should not see hidden messages unless explicitly allowed.
        narrator_canonical = scene.narrator.canonical_name
        branch.context = [
            msg
            for msg in branch.context
            if not msg.get("_hidden")
            or narrator_canonical == (msg.get("_canonical_name") or msg.get("name"))
            or narrator_canonical in set(msg.get("_visible_to") or [])
        ]
        branch.concat_context(loc.scratch_context())
        branch.user_message(
            f"Characters present: {[c.name for c in visible_here]}\n"
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
            print_stream=False,
            name=scene.narrator.name,
        )

        ctx.assistant_message(
            result.content,
            tool_calls=[],
            name=scene.narrator.name,
            reasoning_content=result.reasoning_content,
            canonical_name=scene.narrator.canonical_name,
        )
        if result.content.strip() and scene.narrator.importance >= Importance.IMPORTANT:
            scene.narrator.memory.add_conversation([result.content.strip()])
        return result.content.strip()

    def _build_character_branch(
        self,
        char: Character,
        decision: TurnDecision,
        ctx: ConversationContext | None = None,
        scene: Scene | None = None,
        loc: Location | None = None,
    ) -> ConversationContext:
        """Build the conversation branch that is fed to a character LLM call.

        This mirrors the filtering and prepending that happens inside
        :meth:`_character_turn`, but is exposed so debug tooling can inspect the
        exact message list a character would see.
        """
        ctx = ctx or self._ctx
        scene = scene or self._scene
        loc = loc or self._loc
        if ctx is None or scene is None:
            raise RuntimeError("Engine not started.")
        here_chars = self._here_chars
        b = ctx.branch()
        b.filter_to(char.canonical_name)
        hidden_from_char = _hidden_not_visible(char, here_chars)
        if hidden_from_char:
            b.context = [
                msg for msg in b.context
                if (msg.get("_canonical_name") or msg.get("name")) not in hidden_from_char
            ]
            b.present_entities -= hidden_from_char
        # Use the curated single-assistant view so the character is the only
        # assistant in their own conversation history.
        b.context = b.curated_view(char.name)
        b.head = b.context[-1] if b.context else None
        b.concat_context(char.whoami)
        if char.importance >= Importance.IMPORTANT and loc is not None:
            b.concat_context(loc.scratch_context())
        b.concat_context(char.scene_summary_context)
        b.concat_context(char.status_context)
        if char.inner_log:
            recent = char.inner_log[-3:]
            thoughts = "\n".join(entry.get("inner", "") for entry in recent)
            b.user_message(
                f"Your recent private thoughts:\n{thoughts}",
                name="System",
            )
        skin_desc = char.skin_description()
        if skin_desc:
            b.user_message(
                f"Your current appearance: {skin_desc} (this is your base look; "
                "in-scene modifiers such as dirt or damage may apply).",
                name="System",
            )
        inner_mode = decision.response_mode if decision.response_mode in ("outer_and_inner", "inner_only") else "outer"
        directive_text = f"Current directive: {decision.directive or 'None'}"
        if inner_mode == "outer_and_inner":
            directive_text += (
                "\n\nFor this response, output valid JSON with exactly three keys: "
                "'outer', 'inner', and 'explain'. "
                "'outer' is what other characters perceive. "
                "'inner' is your private thought and must NOT be heard by others. "
                "'explain' is one sentence of reasoning. "
                'Example: {"outer": " spoken line ", "inner": " private thought ", "explain": " why this reaction "}'
            )
        elif inner_mode == "inner_only":
            directive_text += (
                "\n\nFor this response, write ONLY your private inner thought. "
                "Do not speak aloud. No JSON is needed."
            )
        b.user_message(directive_text, name="System")
        return b

    def build_character_context(
        self,
        char: Character,
        decision: TurnDecision | None = None,
    ) -> dict[str, Any]:
        """Return the full LLM context that would be sent to *char* right now.

        Includes the system prompt and the filtered/branched message list.
        """
        if self._scene is None:
            raise RuntimeError("Engine not started.")
        decision = decision or self._last_decision
        if decision is None or decision.next_char != char:
            decision = TurnDecision(
                next_char=char,
                directive="",
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
            )
        branch = self._build_character_branch(char, decision)
        has_tools = char.importance != Importance.ANONYMOUS
        system_prompt = _character_system_prompt(char, self._scene, has_tools=has_tools)
        return {
            "system_prompt": system_prompt,
            "messages": branch.to_list(),
        }

    def _character_turn(
        self,
        scene: Scene,
        ctx: ConversationContext,
        decision: TurnDecision,
        loc: Location,
    ) -> str:
        """Execute an NPC turn.

        The character receives their ``whoami`` as injected context.
        Important characters also receive the location scratch.
        Characters may use tools (recall, wiki_recall, write_scratch, attempt_action) before speaking.

        :return: The character's spoken outer text.
        """
        char = decision.next_char

        # ---- character tools ------------------------------------------------
        recall_tool = tool(
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

        wiki_recall_tool = tool(
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

        attempt_action_tool = tool(
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

        registry = ToolRegistry()
        has_tools = char.importance >= Importance.IMPORTANT

        def _recall_handler(args: str) -> str:
            data = json.loads(args)
            query = data.get("query", "")
            memories = char.memory.recall(
                [query],
                depth="medium",
                client=self.client,
                querier=char,
            )
            if memories:
                return "\n".join(f"- {m}" for m in memories)
            return "You don't recall anything relevant."

        def _wiki_recall_handler(args: str) -> str:
            data = json.loads(args)
            query = data.get("query", "")
            return self.orchestrator.wiki.recall(query, querier=char)

        def _write_scratch_handler(args: str) -> str:
            data = json.loads(args)
            note = data.get("note", "")
            if note:
                if char.scratch.text == "Nothing yet!":
                    char.scratch.text = f"[Note]: {note}"
                else:
                    char.scratch.text += f"\n[Note]: {note}"
            return "Note saved."

        def _attempt_action_handler(args: str) -> str:
            data = json.loads(args)
            data["source"] = char.canonical_name
            self._pending_attempts.append(data)
            return "Action attempt recorded for the orchestrator."

        character_tools: list[dict] = []
        if has_tools:
            registry.register("recall", _recall_handler)
            registry.register("wiki_recall", _wiki_recall_handler)
            registry.register("write_scratch", _write_scratch_handler)
            registry.register("attempt_action", _attempt_action_handler)
            character_tools = [recall_tool, wiki_recall_tool, write_scratch_tool, attempt_action_tool]
        # -------------------------------------------------------------------- #

        inner_mode = decision.response_mode if decision.response_mode in ("outer_and_inner", "inner_only") else "outer"

        system_prompt = _character_system_prompt(char, scene, has_tools=has_tools)

        branch = self._build_character_branch(char, decision, ctx, scene, loc)
        result = self.client.complete(
            role=GameRole.CHARACTER,
            system_prompt=system_prompt,
            messages=branch.to_list(),
            tools=character_tools or None,
            tool_choice="auto" if has_tools else None,
            stream=True,
            print_stream=False,
            name=char.name,
        )

        # Handle tool calls: execute, append to base context, re-call LLM.
        # The model may chain multiple tool calls (e.g. recall → wiki_recall → attempt_action)
        # before finally producing spoken content.
        while result.tool_calls:
            ctx.assistant_message(
                result.content,
                tool_calls=result.tool_calls,
                name=char.name,
                reasoning_content=result.reasoning_content,
                hidden=char.hidden,
                visible_to=set(char.visible_to) if char.hidden else None,
                canonical_name=char.canonical_name,
            )
            for tc in result.tool_calls:
                name = tc["function"]["name"]
                logger.debug(
                    f'Executing character tool call: {char.name}/{name} '
                    f'args={tc["function"]["arguments"]!r}'
                )
                result_text = registry.call(
                    name, tc["function"]["arguments"]
                )
                logger.debug(
                    f'Character tool call {char.name}/{name} returned: {result_text!r}'
                )
                ctx.tool_message(result_text, tool_call_id=tc["id"])

            branch2 = self._build_character_branch(char, decision, ctx, scene, loc)
            result = self.client.complete(
                role=GameRole.CHARACTER,
                system_prompt=system_prompt,
                messages=branch2.to_list(),
                tools=character_tools,
                tool_choice="auto" if has_tools else None,
                stream=True,
                print_stream=False,
                name=char.name,
            )

        outer, inner, explain = _parse_inner_response(result.content, mode=inner_mode)
        if inner is not None:
            char.inner_log.append({
                "outer": outer,
                "inner": inner,
                "explain": explain,
            })
            logger.debug(f"Stored inner monologue for {char.name}: {inner!r}")

        if outer.strip():
            ctx.assistant_message(
                outer,
                tool_calls=[],
                name=char.name,
                reasoning_content=result.reasoning_content,
                hidden=char.hidden,
                visible_to=set(char.visible_to) if char.hidden else None,
                canonical_name=char.canonical_name,
            )
            if char.importance >= Importance.IMPORTANT:
                char.memory.add_conversation([outer.strip()])
        elif inner is not None:
            # Silent thought turn: nothing is spoken aloud.
            logger.debug(f"{char.name} produced an inner-only turn")
        else:
            # Empty or unparseable response: record an empty assistant turn.
            ctx.assistant_message(
                outer,
                tool_calls=[],
                name=char.name,
                reasoning_content=result.reasoning_content,
                canonical_name=char.canonical_name,
            )
        return outer.strip()

    def _apply_decision(self, decision: TurnDecision) -> EngineStepResult:
        """Apply an orchestrator decision and produce a step result."""
        scene = self._scene
        ctx = self._ctx
        here_chars = self._here_chars
        away_chars = self._away_chars
        loc = self._loc

        switch_background = ""

        # Anonymous spawns are normally materialized by the orchestrator when
        # it produces the decision (so its own validation can reference them);
        # those arrive pre-registered in decision.spawned_characters.  Create
        # any remaining ones here as a defensive fallback.
        spawned_canonical: list[str] = list(decision.spawned_characters)
        pre_materialized = set(decision.spawned_characters)
        existing_canonical = {c.canonical_name for c in scene.character_pool}
        for spawn in decision.spawn_anonymous:
            name = spawn.get("name", "")
            if not name:
                logger.warning("Skipping anonymous spawn: no name provided.")
                continue
            if name in existing_canonical:
                if name not in pre_materialized:
                    logger.warning(f"Skipping anonymous spawn for '{name}': already exists.")
                continue
            sprite = spawn.get("sprite", "unknown")
            new_char = create_anonymous_character(
                name,
                description=spawn.get("description", ""),
                sprite=sprite,
                title=spawn.get("title", ""),
            )
            scene.character_pool.add(new_char)
            here_chars.add(new_char)
            ctx.add_entities(new_char.canonical_name)
            spawned_canonical.append(new_char.canonical_name)
            existing_canonical.add(new_char.canonical_name)
        if spawned_canonical:
            logger.info(f"Spawned anonymous characters: {spawned_canonical}")

        # Apply sprite changes from the orchestrator decision.
        # The orchestrator may reference characters by display or canonical name.
        sprite_changes: dict[str, str] = {}
        if decision.change_sprite:
            self._mechanical_changelog.append({
                "turn": self._turn_count,
                "type": "change_sprite",
                "sprites": dict(decision.change_sprite),
            })
        for char_name, sprite_name in decision.change_sprite.items():
            char = scene.character_by_name(char_name)
            if char is not None and char in here_chars:
                if sprite_name == "hidden":
                    char.current_sprite = "none"
                    char.hidden = True
                    char.visible_to = set()
                elif sprite_name == "none":
                    char.current_sprite = "none"
                    char.hidden = False
                    char.visible_to = set()
                else:
                    char.current_sprite = sprite_name
                    char.hidden = False
                    char.visible_to = set()
                sprite_changes[char.canonical_name] = sprite_name

        if decision.next_scene is not None:
            self._next_scene = decision.next_scene
            self._running = False
            return EngineStepResult(
                scene_ended=True, next_scene=decision.next_scene,
                speaker=decision.next_char.canonical_name,
                speaker_title=decision.next_char.title,
                enter=[c.canonical_name for c in decision.entering_chars],
                exit=[c.canonical_name for c in decision.exiting_chars],
                spawn=spawned_canonical,
                sprite_changes=sprite_changes,
                switch_background=switch_background,
                system_changes=decision.system_changes,
                output="",
            )

        self._last_decision = decision
        self._prev_char = decision.next_char

        if decision.next_char != scene.player and decision.directive:
            self._directives_log[decision.next_char] = decision.directive

        if decision.switch_location is not None and decision.switch_location != loc:
            self._mechanical_changelog.append({
                "turn": self._turn_count,
                "type": "switch_location",
                "from": loc.canonical_name if loc else None,
                "to": decision.switch_location.canonical_name,
            })
            self._loc = decision.switch_location
            loc = self._loc

        if decision.switch_background and loc is not None:
            if decision.switch_background in loc.backgrounds:
                self._mechanical_changelog.append({
                    "turn": self._turn_count,
                    "type": "switch_background",
                    "location": loc.canonical_name,
                    "background": decision.switch_background,
                })
                loc.current_background = decision.switch_background
                switch_background = decision.switch_background
            else:
                logger.warning(
                    f"Orchestrator chose invalid background '{decision.switch_background}' "
                    f"for location '{loc.name}'. Valid: {loc.backgrounds}"
                )

        if decision.edit_location and loc is not None:
            self._mechanical_changelog.append({
                "turn": self._turn_count,
                "type": "edit_location",
                "location": loc.canonical_name,
                "edit": decision.edit_location,
            })
            loc.desc += f"\n\n[Update]: {decision.edit_location}"
            logger.info(f"Location '{loc.name}' updated: {decision.edit_location}")

        if decision.set_time:
            self._mechanical_changelog.append({
                "turn": self._turn_count,
                "type": "set_time",
                "time": decision.set_time,
            })
            self._world_time = decision.set_time
            logger.info(f"World time set to: {decision.set_time}")

        if decision.system_changes:
            changes = dict(decision.system_changes)
            target = changes.pop("target", "player")
            target_page, target_label = self._resolve_status_target(target)
            updated = self._apply_page_update(target_page, changes)
            if target == "free":
                self._free_status = updated
            elif target == "player":
                self._player_status = updated
            else:
                loc_target = scene.location_by_name(target)
                if loc_target is not None:
                    loc_target.status = updated
                else:
                    char_target = scene.character_by_name(target)
                    if char_target is not None:
                        char_target.status = updated
            self._mechanical_changelog.append({
                "turn": self._turn_count,
                "type": "system_changes",
                "target": target_label,
                "changes": changes,
            })
            logger.info(f"Status page updated for {target_label}: {list(changes.keys())}")

        enter_names = [c.canonical_name for c in decision.entering_chars]
        exit_names = [c.canonical_name for c in decision.exiting_chars]

        if decision.entering_chars:
            self._mechanical_changelog.append({
                "turn": self._turn_count,
                "type": "enter",
                "chars": [c.canonical_name for c in decision.entering_chars],
            })
            ctx.enter_entities(*enter_names)
            here_chars |= decision.entering_chars
            away_chars -= decision.entering_chars

        if decision.next_char == scene.player:
            self._needs_player_input = True
            return EngineStepResult(
                needs_player_input=True, suggestions=decision.suggestions,
                speaker=decision.next_char.canonical_name,
                speaker_title=decision.next_char.title,
                enter=enter_names,
                exit=exit_names,
                spawn=spawned_canonical,
                sprite_changes=sprite_changes,
                switch_background=switch_background,
                system_changes=decision.system_changes,
                output="",
            )

        if decision.next_char == scene.narrator:
            logger.info(f"{decision.next_char.name} [Narrator] is speaking now.")
            turn_output = self._narrator_turn(scene, ctx, decision, here_chars, away_chars, loc)
            turn_inner = ""
        else:
            logger.info(f"{decision.next_char.name} is speaking now.")
            pre_len = len(decision.next_char.inner_log)
            turn_output = self._character_turn(scene, ctx, decision, loc)
            post = decision.next_char.inner_log
            turn_inner = post[-1].get("inner", "") if len(post) > pre_len else ""

        if decision.exiting_chars:
            self._mechanical_changelog.append({
                "turn": self._turn_count,
                "type": "exit",
                "chars": [c.canonical_name for c in decision.exiting_chars],
            })
            ctx.exit_entities(*exit_names)
            here_chars -= decision.exiting_chars
            away_chars |= decision.exiting_chars

        self._turn_count += 1
        self._speaker_history.append(decision.next_char.canonical_name)

        return EngineStepResult(
            speaker=decision.next_char.canonical_name,
            speaker_title=decision.next_char.title,
            enter=enter_names,
            exit=exit_names,
            spawn=spawned_canonical,
            sprite_changes=sprite_changes,
            switch_background=switch_background,
            system_changes=decision.system_changes,
            output=turn_output,
            inner=turn_inner,
        )

