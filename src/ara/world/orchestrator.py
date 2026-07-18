"""Orchestrator agent that decides who speaks next and how the scene advances."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.tools import ToolRegistry, tool
from ara.llm.models import Context, GameRole
from ara.memory.chroma import ChromaStore
from ara.memory.wiki import WIKI_COLLECTION, WikiStore
from ara.prompts.orchestrator import orchestrator_system_prompt as _orchestrator_system_prompt
from ara.memory.knowledge import Scratchpad
from ara.world.character import Character, create_anonymous_character
from ara.world.fortune_tools import FortuneTools
from ara.world.scene import Location, Scene
from ara.world.system_page import pretty_print
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
    :ivar switch_background: Background stem to activate for the current
        location, or empty string for no change.
    :ivar spawn_anonymous: List of background characters to create on the fly.
    :ivar spawned_characters: Canonical names of anonymous characters the
        orchestrator already materialized while producing this decision (so
        its own validation could reference them).  The engine reports these
        in ``EngineStepResult.spawn`` instead of re-creating them.
    :ivar set_time: New world time to set (e.g. 'night'), or empty string.
    :ivar system_changes: Updates to a status page (player, free, location, or character).
    """

    next_char: Character
    directive: str
    suggestions: list[str]
    entering_chars: set[Character]
    exiting_chars: set[Character]
    switch_location: Location | None
    edit_location: str = ''
    next_scene: str | None = None
    change_sprite: dict[str, str] = field(default_factory=dict)
    switch_background: str = ''
    spawn_anonymous: list[dict[str, str]] = field(default_factory=list)
    spawned_characters: list[str] = field(default_factory=list)
    set_time: str = ''
    system_changes: dict[str, Any] = field(default_factory=dict)
    response_mode: str = 'outer'


class Orchestrator:
    """Builds dynamic tool schemas and calls the LLM to direct scene flow.

    Because the list of valid characters, locations, and scene choices changes
    every turn, the ``next_round`` tool schema is regenerated on each call.
    When strict mode is enabled, the schema includes ``strict: true`` so that
    DeepSeek's beta endpoint guarantees JSON-schema-compliant output,
    eliminating the need for fragile retry loops.
    """

    def __init__(
        self, client: LLMClient, db: ChromaStore | None = None
    ) -> None:
        """Create an orchestrator.

        :param client: LLM client used for orchestration calls.
        :param db: Optional ChromaDB store for wiki recall.
        """
        self.client = client
        self.db = db
        self.wiki = WikiStore(db, client=client)
        self.registry = ToolRegistry()
        self._capture: NextRoundCapture | None = None
        self._spawned_chars: list[dict[str, Any]] = []
        self._system_changes: dict[str, Any] = {}
        self.wiki_collection = WIKI_COLLECTION
        self.prefetched_wiki: str = ''
        self.orchestrator_note: str = ''
        self.scratch = Scratchpad()

    def write_orchestrator_scratch(self, note: str) -> None:
        """Append *note* to the orchestrator's private journal/scratchpad."""
        if not note:
            return
        if self.scratch.text == 'Nothing yet!':
            self.scratch.text = f"[Journal]: {note}"
        else:
            self.scratch.text += f"\n[Journal]: {note}"

    @staticmethod
    def _character_scratch_digest(scene: Scene) -> Context:
        """Return an injected context pair showing the current scene's character scratches.

        This gives the orchestrator director-level sight of character secrets and
        plans without forcing characters to leak them in dialogue.
        """
        lines: list[str] = []
        for char in sorted(scene.character_pool, key=lambda c: c.name):
            text = char.scratch.text
            if text and text != "Nothing yet!":
                lines.append(f"--- {char.name}'s scratch ---\n{text}")
        if not lines:
            return []
        return [
            {
                "role": "user",
                "content": "Private scratchpads of the characters you are directing. Do not reveal these to any character.",
                "name": ConversationContext.default_sysname,
            },
            {
                "role": "assistant",
                "content": "\n\n".join(lines),
                "name": "Orchestrator",
                "_canonical_name": "__orchestrator__",
            },
        ]

    @staticmethod
    def _normalize_wiki_doc(doc: str) -> str:
        """Return a canonical form of a wiki document for deduplication."""
        return WikiStore.normalize_doc(doc)

    def _wiki_recall(
        self,
        query: str,
        n_results: int = 3,
        annotate_trust: bool = False,
        querier: Character | None = None,
        dedup_against_prefetched: bool = False,
        exclude_docs: set[str] | None = None,
        max_distance: float | None = 0.65,
    ) -> str:
        """Search the orchestrator wiki for relevant entries.

        Thin wrapper over :meth:`WikiStore.recall`; see that method for the
        parameter semantics.  When *dedup_against_prefetched* is ``True``,
        entries that already appear in :attr:`prefetched_wiki` are filtered
        out.
        """
        return self.wiki.recall(
            query,
            n_results=n_results,
            annotate_trust=annotate_trust,
            querier=querier,
            dedup_against=(
                getattr(self, 'prefetched_wiki', '')
                if dedup_against_prefetched
                else ''
            ),
            exclude_docs=exclude_docs,
            max_distance=max_distance,
        )

    def prefetch_wiki(
        self,
        query: str,
        n_results: int = 3,
        exclude_docs: set[str] | None = None,
        max_distance: float | None = 0.65,
    ) -> str:
        """Public wrapper for wiki recall used by the summarizer.

        Returns annotated, trust-scored wiki entries for the given query.
        Documents whose normalized content appears in *exclude_docs* are
        filtered out so the summarizer can avoid cross-query duplication.
        Results beyond *max_distance* are discarded as irrelevant.
        """
        return self._wiki_recall(
            query,
            n_results=n_results,
            annotate_trust=True,
            exclude_docs=exclude_docs,
            max_distance=max_distance,
        )

    def _filter_wiki_for_querier(
        self,
        query: str,
        raw_docs: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]],
        querier: Character,
    ) -> list[str]:
        """Run retrieved wiki documents through a querier-aware subagent.

        Thin wrapper over :meth:`WikiStore.filter_for_querier`.
        """
        return self.wiki.filter_for_querier(query, raw_docs, ids, metadatas, querier)

    def _wiki_write(
        self,
        topic: str,
        content: str,
        importance: str = 'notable',
        trust: float = 0.0,
    ) -> str:
        """Write or overwrite a wiki entry.

        Thin wrapper over :meth:`WikiStore.write`.
        """
        return self.wiki.write(topic, content, importance=importance, trust=trust)

    def _wiki_forget(self, topic: str) -> str:
        """Delete a wiki entry.  Thin wrapper over :meth:`WikiStore.forget`."""
        return self.wiki.forget(topic)

    def _system_prompt(
        self, player: Character, narrator: Character, scene: Scene
    ) -> str:
        return _orchestrator_system_prompt(player, narrator, scene)

    def _fallback_decision(
        self,
        scene: Scene,
        here_chars: set[Character],
    ) -> TurnDecision:
        """Return a safe decision when the orchestrator fails to decide."""
        if scene.player in here_chars:
            return TurnDecision(
                next_char=scene.player,
                directive='',
                suggestions=['Continue.', 'What happens next?'],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                next_scene=None,
                switch_background='',
                spawn_anonymous=[],
                set_time='',
                system_changes={},
                response_mode='outer',
            )
        narrator = scene.narrator
        if narrator in here_chars:
            return TurnDecision(
                next_char=narrator,
                directive='Advance the scene with a brief observation.',
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                next_scene=None,
                switch_background='',
                spawn_anonymous=[],
                set_time='',
                system_changes={},
                response_mode='outer',
            )
        fallback = next(iter(here_chars), narrator)
        return TurnDecision(
            next_char=fallback,
            directive='Continue the scene.',
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene=None,
            switch_background='',
            spawn_anonymous=[],
            set_time='',
            system_changes={},
            response_mode='outer',
        )

    def _spawn_anonymous_handler(self, args: str) -> str:
        """Record anonymously spawned characters for this turn."""
        data = json.loads(args)
        chars = (
            data.get('characters', []) if isinstance(data, dict) else data
        )
        if isinstance(chars, dict):
            chars = [chars]
        elif not isinstance(chars, list):
            chars = [chars]
        self._spawned_chars.extend(chars)
        return f'Spawned {len(chars)} anonymous character(s).'

    def _update_status_page_handler(self, args: str) -> str:
        """Store status-page updates for application after the decision."""
        data = json.loads(args)
        if isinstance(data, dict):
            self._system_changes = data
        return 'Status page updated.'

    def _write_orchestrator_scratch_handler(self, args: str) -> str:
        """Append a note to the orchestrator's private journal."""
        data = json.loads(args)
        if isinstance(data, dict):
            note = data.get('note', '')
        else:
            note = str(data)
        self.write_orchestrator_scratch(note)
        return 'Journal entry saved.'

    def _wiki_recall_handler(self, args: str) -> str:
        data = json.loads(args)
        query = data.get('query', '')
        n_results = int(data.get('n_results', 3))
        max_distance = data.get('max_distance')
        if max_distance is not None:
            max_distance = float(max_distance)
        return self._wiki_recall(
            query,
            n_results=n_results,
            dedup_against_prefetched=True,
            max_distance=max_distance,
        )

    def _wiki_write_handler(self, args: str) -> str:
        data = json.loads(args)
        topic = data.get('topic', '')
        content = data.get('content', '')
        importance = data.get('importance', 'notable')
        trust = float(data.get('trust', 0.0))
        if importance == 'trivial':
            return f"Entry '{topic}' was not stored because its importance is trivial."
        return self._wiki_write(
            topic, content, importance=importance, trust=trust
        )

    def _wiki_forget_handler(self, args: str) -> str:
        data = json.loads(args)
        topic = data.get('topic', '')
        return self._wiki_forget(topic)

    @staticmethod
    def _anonymous_sprite_list(scene: Scene) -> str:
        """Discover anonymous sprite pool for on-the-fly spawning.

        Per-story anonymous sprites take priority over the global pool.
        """
        settings = AraSettings()
        story_name = getattr(scene, 'asset_story_name', '')
        story_anonymous_dir = settings.anonymous_path(story_name)
        global_anonymous_dir = settings.anonymous_path()
        anonymous_dirs = [
            d for d in (story_anonymous_dir, global_anonymous_dir) if d.exists()
        ]
        anonymous_sprites = sorted(
            {p.stem for d in anonymous_dirs for p in d.glob('*.png')}
        )
        return ', '.join(anonymous_sprites) if anonymous_sprites else 'none'

    @staticmethod
    def _sprite_info(here_chars: set[Character]) -> str:
        """Build a description of available sprites for the tool prompt."""
        sprite_info_lines = []
        for c in here_chars:
            if c.sprites:
                descs = []
                for s in c.sprites:
                    d = c.skin_description(s)
                    descs.append(f'{s}' + (f' ({d})' if d else ''))
                sprite_info_lines.append(
                    f'  {c.name}: {", ".join(descs)} (current: {c.current_sprite})'
                )
        return '\n'.join(sprite_info_lines) if sprite_info_lines else '  (none)'

    def _build_tools(
        self,
        scene: Scene,
        loc: Location,
        here_chars: set[Character],
        away_chars: set[Character],
        sprite_info: str,
        anonymous_sprite_list: str,
        fortune_tools: FortuneTools,
    ) -> list[dict[str, Any]]:
        """Build the full orchestrator tool schema list for this turn.

        The ``next_round`` schema is regenerated on each call because the
        list of valid characters, locations, and scene choices changes every
        turn.
        """
        list_here = [c.name for c in here_chars]
        list_away = [c.name for c in away_chars]

        # Build valid_next: every present character plus the narrator may act.
        # With reasoning mode enabled, the orchestrator decides when back-to-back
        # turns (e.g. a monologue or continued narration) are appropriate.
        valid_next = [c.name for c in here_chars]
        valid_next.append(scene.narrator.name)

        valid_locs = [loc.name for loc in scene.location_pool]
        valid_scenes = list(scene.next_choices.keys())

        next_round_tool = tool(
            name='next_round',
            description="""Choose the next character/narrator to act, or to end the scene.
If choosing an NPC/the narrator, provide a directive to guide their actions to fulfill the given plot and leave suggestions empty.
If choosing the player, provide suggestions and leave directive empty.
If setting end_scene to True, you can leave everything else blank.
Also decide on what characters enter/exit the scene.
""",
            properties={
                'next_character': {
                    'type': 'string',
                    'enum': valid_next,
                    'description': 'The character to act next. The same character may act again if the flow of the scene calls for it.',
                },
                'directive': {
                    'type': 'string',
                    'description': 'Directive for the chosen character. Omit if the next character is the player.',
                },
                'suggestions': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': 'List of suggestions. Omit if the next character is not the player.',
                },
                'enter_characters': {
                    'type': 'array',
                    'items': {'type': 'string', 'enum': list_away}
                    if list_away
                    else {'type': 'string'},
                    'description': f'List of characters to enter the scene. Valid options: {", ".join(list_away) if list_away else "none"}.',
                },
                'exit_characters': {
                    'type': 'array',
                    'items': {'type': 'string', 'enum': list_here},
                    'description': f'List of characters to exit the scene. Valid options: {", ".join(list_here)}.',
                },
                'switch_location': {
                    'type': 'string',
                    'enum': valid_locs + [''],
                    'description': f'The location to switch to. Current: {loc.name}. Give empty string if no change.',
                },
                'edit_location': {
                    'type': 'string',
                    'description': 'True if characters or the plot alter the environment of the current location. '
                    'Provide a brief description of the modification (this is applied before switch_location). '
                    'Give an empty string if there is no change.',
                },
                'end_scene': {
                    'type': 'boolean',
                    'description': 'Set to true when the scene has reached its conclusion and should end. When true, next_scene MUST be set to a valid follow-up scene.',
                },
                'next_scene': {
                    'type': 'string',
                    'enum': valid_scenes + [''],
                    'description': f'The follow-up scene to transition to. If end_scene is true, this must be a valid scene ID, or an empty string when there are no follow-up scenes (ends the story). Provide empty string if the scene has not ended. Valid scenes: {", ".join(valid_scenes) if valid_scenes else "none"}.'
                },
                'change_sprite': {
                    'type': 'object',
                    'description': f'Optional: change the sprite for one or more on-screen characters. Keys are character names, values are sprite names.\nAvailable sprites:\n{sprite_info}',
                    'additionalProperties': {'type': 'string'},
                },
                'switch_background': {
                    'type': 'string',
                    'description': f'Optional: switch the background image for the current location. Current background: {loc.current_background or "(legacy)"}. Available: {", ".join(loc.backgrounds) if loc.backgrounds else "(legacy)"}. Give empty string for no change.',
                },
                'set_time': {
                    'type': 'string',
                    'description': f'Optional: change the world time (e.g. morning, afternoon, evening, night). Current: {scene.time or "unspecified"}. Give empty string for no change.',
                },
                'response_mode': {
                    'type': 'string',
                    'enum': ['outer', 'outer_and_inner', 'inner_only'],
                    'description': (
                        'Optional: request inner monologue. outer = normal speech; '
                        'outer_and_inner = public line plus private inner thought; '
                        'inner_only = silent thought. Do not use for the narrator. '
                        'Default: outer.'
                    ),
                },
            },
            required=[
                'next_character',
                'enter_characters',
                'exit_characters',
                'switch_location',
                'edit_location',
                'end_scene',
                'next_scene',
            ],
            strict=True,
        )

        spawn_anonymous_tool = tool(
            name='spawn_anonymous',
            description='Spawn one or more background NPCs on the fly. They enter this turn and are present for future turns. Do not spawn characters whose names already exist in the scene.',
            properties={
                'characters': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {
                                'type': 'string',
                                'description': 'Display name for the background character.',
                            },
                            'title': {
                                'type': 'string',
                                'description': 'Optional title or epithet (e.g., "Nuclear Fleet Carrier", "The Corpse-Eater"). Displays as "[title] name" format.',
                            },
                            'description': {
                                'type': 'string',
                                'description': 'What they look like and what they are doing.',
                            },
                            'sprite': {
                                'type': 'string',
                                'description': f'Sprite stem under data/assets/cc/<story>/anonymous/ (or data/assets/cc/anonymous/). Available: {anonymous_sprite_list}. Use "unknown" if unsure.',
                            },
                        },
                        'required': ['name', 'description'],
                    },
                },
            },
            required=['characters'],
            strict=True,
        )

        update_status_page_tool = tool(
            name='update_status_page',
            description="""Update a system/status page (inventory, HP bars, skills, etc.).
Target may be "player" (default), "free" (misc world state), a location name, or a character name.
Legacy form: {"bars": {"HP": 90}, "inventory": ["Key"], "skills": ["Leadership"]}
Preferred DSL form: {"target": "player", "title": "Commander Status", "sections": [{"type": "bars", "items": [{"label": "HP", "value": 90, "max": 100}]}, {"type": "bars", "label": "Team1", "items": [{"label": "Alice", "value": 50}]}, {"type": "bars", "label": "Team2", "items": [{"label": "Bob", "value": 80}]}, {"type": "inventory", "items": [{"id": "bronze_key", "name": "Bronze Key", "description": "Opens the sealed door", "metadata": {"unlocks": "sealed_door"}}]}]}
Inventory items may be strings or {id, name, description, metadata} objects; descriptions are hidden until the player clicks the item. If an item has an "id" matching a plot item template, the engine will fill missing name/description/metadata from the template.
Supported section types: bars, inventory, skills, text. Add a "label" field to sections when you need multiple sections of the same type (e.g., separate HP bars for multiple teams). Without a label, a section replaces any previous section of the same type.""",
            properties={
                'target': {
                    'type': 'string',
                    'description': 'Target status page: "player", "free", a location name, or a character name. Defaults to "player".',
                },
                'title': {'type': 'string'},
                'sections': {'type': 'array', 'description': 'Section objects with keys: type, label (optional, to prevent overwriting same-type sections), items.', 'items': {'type': 'object'}},
                'bars': {
                    'type': 'object',
                    'additionalProperties': {'type': 'number'},
                },
                'inventory': {'type': 'array', 'items': {'type': 'string'}},
                'skills': {'type': 'array', 'items': {'type': 'string'}},
            },
            required=[],
            strict=False,
        )

        write_orchestrator_scratch_tool = tool(
            name='write_orchestrator_scratch',
            description="""Append a private note to your journal/scratchpad.
Use this to record facts, plans, reminders, or ongoing concerns that should persist across turns within the scene and carry over to the next scene. This is visible only to you.
""",
            properties={
                'note': {
                    'type': 'string',
                    'description': 'The journal entry to append. Keep it concise and useful for future turns.',
                },
            },
            required=['note'],
            strict=True,
        )

        wiki_recall_tool = tool(
            name='wiki_recall',
            description=(
                'Search the permanent wiki for important facts, relationships, or rules. '
                'Use this when the current situation may depend on established lore.'
            ),
            properties={
                'query': {
                    'type': 'string',
                    'description': 'Search query describing what you want to remember.',
                },
                'n_results': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 10,
                    'description': 'Number of entries to retrieve (default 3).',
                },
                'max_distance': {
                    'type': 'number',
                    'minimum': 0.0,
                    'maximum': 2.0,
                    'description': (
                        'Maximum semantic distance for a result to be considered relevant. '
                        'Farther results are discarded. Default 0.65.'
                    ),
                },
            },
            required=['query'],
            strict=True,
        )

        wiki_write_tool = tool(
            name='wiki_write',
            description=(
                'Write an important fact to the permanent wiki. Only store information '
                'that meaningfully changes the state of the world, relationships, or plot. '
                'Do not store routine dialogue, transient emotions, or color commentary.'
            ),
            properties={
                'topic': {
                    'type': 'string',
                    'description': 'Short unique topic or title for this entry.',
                },
                'content': {
                    'type': 'string',
                    'description': 'The important fact to remember.',
                },
                'importance': {
                    'type': 'string',
                    'enum': ['trivial', 'notable', 'important', 'critical'],
                    'description': 'How significant this fact is. Trivial entries may be ignored.',
                },
                'trust': {
                    'type': 'number',
                    'minimum': -1.0,
                    'maximum': 1.0,
                    'description': (
                        'Trustworthiness of this fact: 1.0 is established canon, '
                        '0.5 is a plausible rumor, 0.0 is invented on the spot, '
                        '-1.0 is explicitly a lie. Defaults to 0.0.'
                    ),
                },
            },
            required=['topic', 'content', 'importance'],
            strict=True,
        )

        wiki_forget_tool = tool(
            name='wiki_forget',
            description='Delete a wiki entry by topic.',
            properties={
                'topic': {
                    'type': 'string',
                    'description': 'Topic of the entry to delete.',
                },
            },
            required=['topic'],
            strict=True,
        )

        return [
            next_round_tool,
            spawn_anonymous_tool,
            update_status_page_tool,
            write_orchestrator_scratch_tool,
            *fortune_tools.tools(),
            wiki_recall_tool,
            wiki_write_tool,
            wiki_forget_tool,
        ]

    def _prepare_turn_branch(
        self,
        ctx: ConversationContext,
        scene: Scene,
        here_chars: set[Character],
        away_chars: set[Character],
        speaker_history: list[str] | None,
        story_state: dict[str, Any] | None,
        player_status: dict[str, Any] | None,
        free_status: dict[str, Any] | None,
        attempts_for_orchestrator: list[dict[str, Any]] | None,
        history: str,
        turn_count: int,
        sprite_info: str,
    ) -> ConversationContext:
        """Assemble the per-turn prompt branch for the orchestrator call.

        Injects past-scene history, plot info, dynamic scene state, the
        orchestrator journal, and character scratch digests into a branch of
        *ctx* and returns it.
        """
        list_here_display = [c.display_name_with_title() for c in here_chars]
        list_away_display = [c.display_name_with_title() for c in away_chars]

        speaker_history = speaker_history or []
        recent = speaker_history[-3:] if speaker_history else []
        story_state = dict(story_state) if story_state else {}

        recent_speakers_str = ', '.join(recent) if recent else '(none)'

        # Build per-character briefing including prev-scene summaries.
        char_info_lines: list[str] = []
        for c in here_chars:
            parts = [
                f'{c.name} ({c.importance.name})',
                f'personality: {c.card_fields.get("personality", "")}',
                f'scenario: {c.card_fields.get("scenario", "")}',
            ]
            if c.title:
                parts.append(f'title: {c.title}')
            if c.hidden:
                visible = ', '.join(sorted(c.visible_to)) or 'none'
                parts.append(f'[hidden; visible to: {visible}]')
            if c.prev_scene_summary:
                parts.append(f'orientation: {c.prev_scene_summary}')
            char_info_lines.append(', '.join(parts))

        away_info_lines: list[str] = []
        for c in away_chars:
            parts = [f'{c.name} ({c.importance.name})']
            if c.title:
                parts.append(f'title: {c.title}')
            if c.prev_scene_summary:
                parts.append(f'orientation: {c.prev_scene_summary}')
            away_info_lines.append(', '.join(parts))

        status_lines: list[str] = []
        player_status = pretty_print(player_status)
        if player_status:
            status_lines.append(f'Player status:\n{player_status}')
        free_status_text = pretty_print(free_status)
        if free_status_text:
            status_lines.append(f'Free/world status:\n{free_status_text}')
        for location in scene.location_pool:
            loc_status = pretty_print(location.status)
            if loc_status:
                status_lines.append(
                    f"Location '{location.name}' status:\n{loc_status}"
                )
        for c in here_chars | away_chars | {scene.narrator}:
            char_status = pretty_print(c.status)
            if char_status:
                status_lines.append(
                    f"Character '{c.name}' status:\n{char_status}"
                )
        status_section = '\n\n'.join(status_lines)

        prefetched_wiki = getattr(self, 'prefetched_wiki', '')
        wiki_section = (
            f'\n\nRelevant wiki entries:\n{prefetched_wiki}\n'
            if prefetched_wiki and 'No relevant' not in prefetched_wiki
            else ''
        )

        orchestrator_note = getattr(self, 'orchestrator_note', '')
        self.orchestrator_note = ''
        note_section = (
            f'\n\nNote from previous scene:\n{orchestrator_note}\n'
            if orchestrator_note
            else ''
        )

        story_state_section = ''
        if story_state:
            story_state_section = f"""\nStory state (persistent narrative flags):
{story_state}\n"""

        plot_content = f"""Plot:
{scene.plot_as_tool_content()}

Characters currently here: {list_here_display}
Info:
{'\\n'.join(char_info_lines)}

Available sprites per character:
{sprite_info}

Characters currently away: {list_away_display}
{'\\n'.join(away_info_lines) if away_info_lines else '(no away characters with orientation)'}

Note: ANONYMOUS characters are background extras with minimal persistence. They do not have detailed backstories or memory.
"""

        attempts_section = ''
        if attempts_for_orchestrator:
            lines = []
            for attempt in attempts_for_orchestrator:
                source = attempt.get('source', 'Unknown')
                action = attempt.get('action', '')
                intent = attempt.get('intent', '')
                target = attempt.get('target', '')
                secrecy = attempt.get('secrecy', '')
                parts = [f'{source}: {action}']
                if intent:
                    parts.append(f'intent: {intent}')
                if target:
                    parts.append(f'target: {target}')
                if secrecy:
                    parts.append(f'secrecy: {secrecy}')
                lines.append(' - ' + '; '.join(parts))
            attempts_section = '\nPending action attempts:\n' + '\n'.join(lines)

        status_part = f'\n\n{status_section}' if status_section else ''
        dynamic_state = f"""Turn count: {turn_count}
Recent speakers: {recent_speakers_str}
{wiki_section}{note_section}{story_state_section}{attempts_section}{status_part}""".strip()

        branch = ctx.branch()
        if branch.head is not None and branch.head.get('role') == 'assistant':
            branch.user_message('Continue.', name='System')

        if history and isinstance(history, str):
            branch.user_message(
                'The following message containes past scene information.',
                name='System',
            )
            branch.assistant_message(str(history), name='System')

        branch.user_message(
            'The following message contains the current plot information.',
            name='System',
        )
        branch.assistant_message(str(plot_content), name='System')

        if dynamic_state:
            branch.user_message(
                'The following message contains the current dynamic scene state.\n\n'
                + dynamic_state,
                name='System',
            )

        if self.scratch.text and self.scratch.text != 'Nothing yet!':
            branch.user_message(
                'Your private journal (only you see this).',
                name='System',
            )
            branch.assistant_message(
                self.scratch.text,
                tool_calls=[],
                name='Journal',
                canonical_name='__orchestrator__',
            )

        branch.concat_context(self._character_scratch_digest(scene))

        # Keep each source message in its own user message so the per-turn
        # KV-cache prefix stays stable instead of growing one giant merged user
        # block every round.
        branch.context = branch.curated_view('__orchestrator__', collapse=False)
        branch.head = branch.context[-1] if branch.context else None
        return branch

    def decide_next_turn(
        self,
        scene: Scene,
        ctx: ConversationContext,
        here_chars: set[Character],
        away_chars: set[Character],
        prev_char: Character | None,
        loc: Location,
        history: str = '',
        turn_count: int = 0,
        speaker_history: list[str] | None = None,
        story_state: dict[str, Any] | None = None,
        attempts_for_orchestrator: list[dict[str, Any]] | None = None,
        player_status: dict[str, Any] | None = None,
        free_status: dict[str, Any] | None = None,
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
        self._capture = NextRoundCapture()
        self._capture._data = None
        self.registry.register('next_round', self._capture.hook)
        self._spawned_chars = []
        self._system_changes = {}

        self.registry.register('spawn_anonymous', self._spawn_anonymous_handler)
        self.registry.register(
            'update_status_page', self._update_status_page_handler
        )
        self.registry.register(
            'write_orchestrator_scratch', self._write_orchestrator_scratch_handler
        )

        story_name = getattr(scene, 'asset_story_name', None) or None
        fortune_tools = FortuneTools(story=story_name)
        fortune_tools.register(self.registry)

        self.registry.register('wiki_recall', self._wiki_recall_handler)
        self.registry.register('wiki_write', self._wiki_write_handler)
        self.registry.register('wiki_forget', self._wiki_forget_handler)

        anonymous_sprite_list = self._anonymous_sprite_list(scene)
        sprite_info = self._sprite_info(here_chars)

        tools = self._build_tools(
            scene, loc, here_chars, away_chars,
            sprite_info, anonymous_sprite_list, fortune_tools,
        )
        branch = self._prepare_turn_branch(
            ctx, scene, here_chars, away_chars,
            speaker_history, story_state, player_status, free_status,
            attempts_for_orchestrator, history, turn_count, sprite_info,
        )

        logger.debug('Control handed to orchestrator')
        result = self.client.complete(
            role=GameRole.ORCHESTRATOR,
            system_prompt=self._system_prompt(
                scene.player, scene.narrator, scene
            ),
            messages=branch.to_list(),
            tools=tools,
            stream=True,
            print_stream=False,
        )

        # Handle tool-call loops: roll/random may be called before next_round.
        max_tool_loops = 20
        for _ in range(max_tool_loops):
            if not result.tool_calls:
                logger.warning(
                    f'Orchestrator returned no tool calls (content={result.content!r}). '
                    f'Retries remaining: {_retries}'
                )
                if _retries > 0:
                    branch.user_message(
                        'SYSTEM WARNING: You failed to call any tool. '
                        'You MUST call the next_round tool to advance the scene. '
                        'Do NOT output narration, dialogue, or prose. '
                        'Your ONLY valid response is a tool call.',
                        name='System',
                    )
                    result = self.client.complete(
                        role=GameRole.ORCHESTRATOR,
                        system_prompt=self._system_prompt(
                            scene.player, scene.narrator, scene
                        ),
                        messages=branch.to_list(),
                        tools=tools,
                        stream=True,
                        print_stream=False,
                    )
                    _retries -= 1
                    continue
                raise RuntimeError(
                    'Orchestrator failed to produce a tool call after retries.'
                )

            try:
                # Record the assistant's tool-call message so that subsequent
                # tool-result messages are valid for the API.
                branch.assistant_message(
                    result.content,
                    tool_calls=result.tool_calls,
                    reasoning_content=result.reasoning_content,
                    canonical_name='__orchestrator__',
                )

                has_decision = False
                tool_results: list[tuple[str, str]] = []
                for tc in result.tool_calls:
                    name = tc['function']['name']
                    logger.debug(
                        f'Executing orchestrator tool call: {name} '
                        f'args={tc["function"]["arguments"]!r}'
                    )
                    if name == 'next_round':
                        has_decision = True
                    result_text = self.registry.call(
                        name, tc['function']['arguments']
                    )
                    logger.debug(
                        f'Orchestrator tool call {name} returned: {result_text!r}'
                    )
                    tool_results.append((tc['id'], result_text))

                if has_decision:
                    break

                # roll/random were called; append results and re-call the LLM.
                for tool_call_id, result_text in tool_results:
                    branch.tool_message(result_text, tool_call_id=tool_call_id)

                result = self.client.complete(
                    role=GameRole.ORCHESTRATOR,
                    system_prompt=self._system_prompt(
                        scene.player, scene.narrator, scene
                    ),
                    messages=branch.to_list(),
                    tools=tools,
                    stream=True,
                    print_stream=False,
                )
            except json.JSONDecodeError as exc:
                logger.warning(
                    f'Orchestrator returned malformed JSON ({exc!r}). '
                    f'Retries remaining: {_retries}'
                )
                # The assistant tool-call message is already on the branch, but
                # we failed to execute the call(s). Emit error tool-result
                # messages so the conversation remains API-valid before retrying.
                for tc in result.tool_calls:
                    branch.tool_message(
                        f'SYSTEM ERROR: malformed JSON arguments for '
                        f"{tc['function']['name']}: {exc}",
                        tool_call_id=tc['id'],
                    )
                if _retries > 0:
                    branch.user_message(
                        'SYSTEM WARNING: Your previous tool call contained '
                        'malformed JSON. Ensure all JSON arguments are properly '
                        'escaped and structured. Call the next_round tool with '
                        'correctly formatted arguments.',
                        name='System',
                    )
                    result = self.client.complete(
                        role=GameRole.ORCHESTRATOR,
                        system_prompt=self._system_prompt(
                            scene.player, scene.narrator, scene
                        ),
                        messages=branch.to_list(),
                        tools=tools,
                        stream=True,
                        print_stream=False,
                    )
                    _retries -= 1
                    continue
                raise RuntimeError(
                    'Orchestrator failed to produce valid JSON after retries.'
                )
        else:
            if self._capture._data is None:
                logger.warning(
                    'Orchestrator kept calling auxiliary tools without deciding; '
                    'using fallback decision.'
                )
                return self._fallback_decision(scene, here_chars)
            logger.warning(
                'Orchestrator kept calling auxiliary tools without deciding; '
                'returning best effort decision.'
            )

        assert self._capture is not None
        # If the orchestrator spawned anonymous characters this turn, materialize
        # them now so that next_round can select one of them as the next speaker
        # and so that exit/enter/sprite fields can reference them.
        spawned_chars: set[Character] = set()
        existing_names = {c.canonical_name for c in scene.character_pool}
        for spawn in self._spawned_chars:
            name = spawn.get("name", "")
            if not name or name in existing_names:
                continue
            new_char = create_anonymous_character(
                name,
                description=spawn.get("description", ""),
                sprite=spawn.get("sprite", ""),
                title=spawn.get("title", ""),
            )
            scene.character_pool.add(new_char)
            here_chars.add(new_char)
            ctx.add_entities(new_char.canonical_name)
            spawned_chars.add(new_char)
            existing_names.add(name)

        # Ensure freshly spawned anonymous characters are selectable as the
        # next speaker and can be referenced in enter/exit lists.
        spawn_chars = {
            scene.character_by_name(s.get('name'))
            for s in self._spawned_chars
            if scene.character_by_name(s.get('name'))
        }
        decision = self._capture.to_decision(
            here_chars | {scene.narrator} | spawn_chars,
            away_chars,
            scene.location_pool,
            scene,
        )
        decision.spawn_anonymous = self._spawned_chars
        decision.spawned_characters = [c.canonical_name for c in spawned_chars]
        decision.system_changes = self._system_changes
        switch_name = (
            decision.switch_location.name
            if decision.switch_location
            else '(none)'
        )
        attempts = attempts_for_orchestrator or []
        attempt_details = ''
        if attempts:
            attempt_summaries = []
            for attempt in attempts:
                source = attempt.get('source', 'Unknown')
                action = attempt.get('action', str(attempt))
                attempt_summaries.append(f'{source}: {action}')
            attempt_details = ' [' + '; '.join(attempt_summaries) + ']'
        logger.info(
            f'Orchestrator decision: scene={scene.id}, loc={loc.name}, '
            f'next={decision.next_char.name}, '
            f'directive={decision.directive!r}, '
            f'suggestions={decision.suggestions!r}, '
            f'enter={[c.name for c in decision.entering_chars]}, '
            f'exit={[c.name for c in decision.exiting_chars]}, '
            f'switch_location={switch_name}, '
            f'edit_location={decision.edit_location!r}, '
            f'end_scene={decision.next_scene is not None}, '
            f'next_scene={decision.next_scene}, '
            f'change_sprite={decision.change_sprite}, '
            f'response_mode={decision.response_mode}, '
            f'attempts={len(attempts)}{attempt_details}'
        )
        return decision


class NextRoundCapture:
    """Internal helper that captures the arguments of the ``next_round`` tool."""

    def __init__(self) -> None:
        """Create a fresh capture with no decision stored yet."""
        self._data: dict | None = None

    def hook(self, args: str) -> str:
        """Parse and store the JSON arguments.

        :param args: JSON-encoded argument string.
        :return: Empty string (tool result is discarded for the orchestrator).
        """
        self._data = json.loads(args)
        return ''

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
            raise RuntimeError('Orchestrator did not produce a decision.')

        end_scene = self._data.get('end_scene', False)
        next_scene = self._data.get('next_scene', '')

        if end_scene and not next_scene:
            valid_scenes = list(scene.next_choices.keys())
            if valid_scenes:
                next_scene = valid_scenes[0]
                logger.warning(
                    f'Orchestrator set end_scene=true but next_scene was empty. '
                    f"Auto-selected '{next_scene}'."
                )
            else:
                # No follow-up scene is defined; end the story.
                next_scene = ""

        if end_scene or next_scene:
            if next_scene and next_scene not in scene.next_choices:
                raise RuntimeError(
                    f"Invalid next_scene '{next_scene}'. "
                    f'Valid options: {list(scene.next_choices.keys())}'
                )
            return TurnDecision(
                next_char=scene.player,
                directive='',
                suggestions=[],
                entering_chars=set(),
                exiting_chars=set(),
                switch_location=None,
                next_scene=next_scene,
                switch_background='',
                spawn_anonymous=[],
                set_time='',
                system_changes={},
            )

        # Validate and resolve change_sprite. The LLM may use display or
        # canonical names; resolve either to the runtime character.
        change_sprite: dict[str, str] = {}
        raw_changes = self._data.get('change_sprite', {})
        if isinstance(raw_changes, dict):
            for char_name, sprite_name in raw_changes.items():
                char = scene.character_by_name(char_name)
                if char is None or char not in present_chars:
                    logger.warning(
                        f"Orchestrator tried to change sprite for unknown character '{char_name}'"
                    )
                    continue
                if (
                    sprite_name not in ('none', 'hidden')
                    and sprite_name not in char.sprites
                ):
                    logger.warning(
                        f"Orchestrator chose invalid sprite '{sprite_name}' for {char.name}. "
                        f'Valid: {char.sprites}'
                    )
                    continue
                change_sprite[char.canonical_name] = sprite_name

        next_name = self._data['next_character']
        next_char = scene.character_by_name(next_name)
        if next_char is None or next_char not in present_chars:
            raise RuntimeError(
                f"Character '{next_name}' not found. "
                f'Valid options: {[c.name for c in present_chars]}'
            )

        response_mode = self._data.get('response_mode', 'outer')
        if response_mode not in ('outer', 'outer_and_inner', 'inner_only'):
            response_mode = 'outer'
        if next_char == scene.narrator:
            response_mode = 'outer'

        def _find(names: list[str], pool: set[Character]) -> set[Character]:
            result: set[Character] = set()
            for n in names:
                char = scene.character_by_name(n)
                if char and char in pool:
                    result.add(char)
                else:
                    logger.warning(
                        f"Orchestrator named unknown character '{n}' in "
                        f'enter/exit list. Valid options: {[c.name for c in pool]}'
                    )
            return result

        switch_name = self._data.get('switch_location', '')
        switch_loc = (
            scene.location_by_name(switch_name) if switch_name else None
        )
        if switch_name and switch_loc is None:
            raise RuntimeError(
                f"Location '{switch_name}' not found. "
                f'Valid options: {[l.name for l in loc_pool]}'
            )

        return TurnDecision(
            next_char=next_char,
            directive=self._data.get('directive', ''),
            suggestions=self._data.get('suggestions', []),
            entering_chars=_find(
                self._data.get('enter_characters', []), away_chars
            ),
            exiting_chars=_find(
                self._data.get('exit_characters', []), present_chars
            ),
            switch_location=switch_loc,
            edit_location=self._data.get('edit_location', ''),
            next_scene=None,
            change_sprite=change_sprite,
            switch_background=self._data.get('switch_background', ''),
            spawn_anonymous=[],
            set_time=self._data.get('set_time', ''),
            system_changes={},
            response_mode=response_mode,
        )
