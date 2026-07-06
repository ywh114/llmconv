"""Orchestrator agent that decides who speaks next and how the scene advances."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.tools import ToolRegistry, tool
from ara.llm.models import Context, GameRole
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import Scratchpad
from ara.world.character import Character, create_anonymous_character
from ara.world.scene import Location, Scene
from ara.world.system_page import pretty_print
from ara.world import fortune
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
        self.registry = ToolRegistry()
        self._capture: _NextRoundCapture | None = None
        self._spawned_chars: list[dict[str, Any]] = []
        self._system_changes: dict[str, Any] = {}
        self.wiki_collection = 'orchestrator_wiki'
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
        text = doc.strip()
        if text.startswith('-'):
            text = text[1:].strip()
        if text.startswith('(trust:') and ')' in text:
            text = text.split(')', 1)[1].strip()
        return ' '.join(text.split())

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

        :param query: Search query.
        :param n_results: Maximum number of entries to retrieve.
        :param annotate_trust: When ``True``, prefix each result with its trust score.
        :param querier: Optional character requesting the information.  When
            provided, a filtering subagent reframes the result for the querier's
            perspective and expertise.
        :param dedup_against_prefetched: When ``True``, filter out entries that
            already appear in :attr:`prefetched_wiki`.
        :param exclude_docs: Optional set of normalized document contents to
            omit from the result.  Used by the summarizer to avoid retrieving
            the same wiki entry for multiple prefetch queries.
        :param max_distance: Maximum ChromaDB distance for a result to be
            considered relevant.  Results farther than this are discarded.
        :return: Formatted wiki results or a failure message.
        """
        if self.db is None:
            return 'Wiki is not available.'
        logger.debug(f'Wiki recall query: {query!r}')
        try:
            results = self.db.query(
                self.wiki_collection,
                query_texts=[query],
                n_results=n_results,
            )
            docs = results.get('documents', [[]])[0] or []
            if not docs:
                return 'No relevant wiki entries found.'

            ids = results.get('ids', [[]])[0] or [
                str(i) for i in range(len(docs))
            ]
            metadatas = results.get('metadatas', [[]])[0] or [{} for _ in docs]
            distances = results.get('distances', [[]])[0] or []

            if max_distance is not None and distances:
                filtered = [
                    (d, i, m)
                    for d, i, m, dist in zip(docs, ids, metadatas, distances)
                    if dist <= max_distance
                ]
                if not filtered:
                    return 'No relevant wiki entries found.'
                docs, ids, metadatas = [list(x) for x in zip(*filtered)]

            items = list(zip(docs, ids, metadatas))
            if exclude_docs:
                items = [
                    (d, i, m) for d, i, m in items
                    if self._normalize_wiki_doc(d) not in exclude_docs
                ]
                if not items:
                    return 'All results already covered by existing context.'
            docs, ids, metadatas = [list(x) for x in zip(*items)]

            if annotate_trust or querier is not None:
                if annotate_trust:
                    formatted: list[str] = []
                    for doc, meta in zip(docs, metadatas):
                        trust = (
                            meta.get('trust')
                            if isinstance(meta, dict)
                            else None
                        )
                        if trust is not None:
                            formatted.append(f'(trust: {trust}) {doc}')
                        else:
                            formatted.append(f'{doc}')
                    docs = formatted

                if querier is not None:
                    docs = self._filter_wiki_for_querier(
                        query, docs, ids, metadatas, querier
                    )

            if dedup_against_prefetched:
                prefetched = getattr(self, 'prefetched_wiki', '')
                if prefetched:
                    prefetched_docs = {
                        self._normalize_wiki_doc(chunk)
                        for chunk in prefetched.split('\n\n')
                        if chunk.strip()
                    }
                    docs = [
                        d for d in docs
                        if self._normalize_wiki_doc(d) not in prefetched_docs
                    ]

            if not docs:
                return 'All results already covered by existing context.'

            text = '\n\n'.join(f'- {d}' for d in docs)
            logger.debug(f'Wiki recall result ({len(docs)} docs):\n{text}')
            return text
        except Exception as exc:
            logger.debug(f'Wiki recall failed: {exc}')
            return 'Wiki recall failed.'

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

        The subagent reframes facts (or invents consistent answers when no raw
        document matches) based on the querier's identity, age, and expertise.
        Topic IDs and source-world metadata are provided so the subagent can
        resolve conflicts between the initial world setting and later settings.
        """
        if not raw_docs:
            return raw_docs
        doc_lines = []
        for doc_id, meta, doc in zip(ids, metadatas, raw_docs):
            world = (
                meta.get('world', 'unknown')
                if isinstance(meta, dict)
                else 'unknown'
            )
            topic = (
                meta.get('topic', doc_id) if isinstance(meta, dict) else doc_id
            )
            doc_lines.append(f'[{topic} | world: {world}] {doc}')
        context = (
            f'Querier: {querier.name}\n'
            f'Personality/expertise: {querier.card_fields.get("personality", "")}\n'
            f'Scenario/background: {querier.card_fields.get("scenario", "")}\n'
            f'Original query: {query}\n\n'
            f'Raw wiki documents (topic ID and source world shown in brackets):\n'
            + '\n'.join(f'- {line}' for line in doc_lines)
        )
        filtered = self.client.complete_subagent(
            task=(
                'Reframe the raw wiki documents into an answer for the querier. '
                'If the querier is a child, simplify. '
                'If the querier is an expert, be detailed and precise. '
                'You must base your answer ONLY on the raw wiki documents provided. '
                'If documents conflict (for example, the initial world setting versus '
                'a mid-game setting), use the topic ID and source world metadata to '
                'decide which fact applies to the current query context. '
                "If no document answers the query, return exactly '-- nothing relevant found'. "
                'Do not invent, infer, or hallucinate facts that are not present. '
                'Return a short paragraph.'
            ),
            context=context,
            max_tokens=256,
        )
        if filtered.strip():
            return [filtered.strip()]
        return raw_docs

    def _wiki_write(
        self,
        topic: str,
        content: str,
        importance: str = 'notable',
        trust: float = 0.0,
    ) -> str:
        """Write or overwrite a wiki entry.

        :param trust: Reliability score from -1.0 (explicit lie) to 1.0 (canon).
        """
        if self.db is None:
            return 'Wiki is not available.'
        try:
            self.db.upsert(
                self.wiki_collection,
                ids=[topic],
                documents=[content],
                metadatas=[
                    {'topic': topic, 'importance': importance, 'trust': trust}
                ],
            )
            result = f"Wiki entry '{topic}' saved."
            logger.debug(f'Wiki write: {result}')
            return result
        except Exception as exc:
            logger.debug(f'Wiki write failed: {exc}')
            return 'Wiki write failed.'

    def _wiki_forget(self, topic: str) -> str:
        """Delete a wiki entry."""
        if self.db is None:
            return 'Wiki is not available.'
        try:
            collection = self.db.collection(self.wiki_collection)
            collection.delete(ids=[topic])
            result = f"Wiki entry '{topic}' deleted."
            logger.debug(f'Wiki forget: {result}')
            return result
        except Exception as exc:
            logger.debug(f'Wiki forget failed: {exc}')
            return 'Wiki forget failed.'

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

All scene history, plot information, and character speech are shown as user messages such as \"Alice says: ...\" or \"Alice attempts recall\". Your own tool calls and reasoning are the only assistant turns. You are the only assistant in this conversation.

Your goal is to steer the player and characters through the story, not to force them.
Player inputs fall into two paths:
- `attempt` (nontrivial physical actions): resolve these in-world using fortune tools, character judgement, and real consequences. A silly attempt still gets a serious resolution; a serious attempt also gets a serious resolution. Do not block attempts.
- `reply` (direct player response/speech): honor this. A silly response still happens; keep the rest of the world canonical and have NPCs react in character.
Playing along means using reply for speech and trivial acts while letting orchestrator decide success for nontrivial acts ussing attempt.
The player has ultimate freedom. If they play along, they should not be able to tell they are being steered through the plot.
If exploration outside the plot happens, keep it canonical with the setting/wiki and gently steer back — NPC reminders are fine, but "the world being against them" is also a valid way to nudge them toward the plot.
DO NOT add extraneous events just to fill time.

## ABSOLUTE RULES - DO NOT VIOLATE
- **NEVER roleplay. NEVER speak in character. NEVER output dialogue, narration, or prose.**
  Your ONLY output is a tool call. You are a machine that decides who speaks next.
- **NEVER act as the narrator.** The narrator is a separate character ({narrator.name}).
  You merely decide WHEN the narrator speaks; you do NOT write their lines.
- **You MUST call the `next_round` tool on EVERY single turn.**
  No exceptions. No free text. No thinking out loud. Just the tool call.
- **NEVER give a directive to the player character.**
   If `next_character` is the player ({player.name}), `directive` MUST be empty and you MUST provide `suggestions`.
   If `next_character` is anyone else, `suggestions` MUST be empty and you MUST provide a `directive`.
- **NEVER give the player ({player.name}) back-to-back turns.** After the player speaks or acts, at least one NPC or the narrator MUST take a turn before the player can act again.

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
    - **One public beat per turn.** Do not use the narrator to compress multiple plot points into a single sentence. If something important happens, let the relevant character speak or act on their turn.
    - **If you make a character enter the scene, do not let the narrator merely mention them and move on.** Have a present character react to the entrance on the next turn, then let the entered character speak or act on the turn after that. The player must actually see important arrivals unfold.
    - **Player attempts must be shown.** After resolving an `attempt` with fortune/status tools, choose a speaker whose next turn will show the outcome to the player (the acting character, a witness, or the narrator describing the visible result). Do not skip the public description.
    - End the scene IMMEDIATELY when the plot's conclusion is reached. Do NOT add a closing narration before ending.
    - When ending the scene, set `end_scene` to `true` and `next_scene` to the most appropriate follow-up scene.
    - When `end_scene` is `true`, `next_scene` MUST be a valid scene ID (not empty).
    - If the turn count is high compared to plot length, wrap up the scene quickly and end it.

3. **Tool instructions**
    - You MUST use the `next_round` tool on EVERY turn. Do NOT output free text. Always call the tool.
    - Use the next_character field to specify the next character.
    - Use the `directive` field ONLY for NPCs and the narrator. It MUST be empty when `next_character` is the player.
    - Use the `suggestions` field ONLY when `next_character` is the player. It MUST be empty for NPCs and the narrator.
    - Use the response_mode field to request inner monologue. Valid values:
      - `outer`: normal speech/action (default).
      - `outer_and_inner`: public line plus a private inner thought.
      - `inner_only`: silent thought turn with no public speech.
      Do not use response_mode for the narrator. Inner thoughts are visible to you (the orchestrator) and may be shared with other characters when the plot/character descriptions justify it.
    - Entering characters enter at the start of the current round of conversation. They MAY BE the next speaker, allowing them to speak or act immediately upon arrival.
    - Exiting characters exit at the end of the current round of conversation. They MAY still be the next speaker, speaking or acting just before they leave.
    - Use `spawn_anonymous` to add one or more background NPCs to the scene on the fly.
    - Use `update_status_page` to change inventory, HP bars, skills, or other system-page state. You may target `"player"` (default), `"free"`, a location by name, or a character by name. Inventory items may include an optional `metadata` dict for plot-relevant behavior (e.g. `{{"unlocks": "sealed_door"}}`).
    - Use `write_orchestrator_scratch(note)` to append a private journal entry. Use it to track facts, plans, unresolved questions, or anything you want to remember across turns. The journal is visible only to you and carries over into the next scene.
    - Use `fortune_roll(n, m)` to resolve uncertain actions with objective randomness (e.g., roll 2d100 to see if a risky action succeeds). **The result of a fortune_roll is BINDING. You must always respect the pass/fail outcome and the stated pass_result or fail_result text. Never override or ignore a roll because it contradicts narrative preference — the dice govern, not dramatic convenience.**
    - Use `fortune_random(distrib, ...)` to inject weighted randomness into pacing or minor outcomes.
    - Use `fortune_iching()` to cast an I-Ching hexagram for omens and general direction.
    - Use `fortune_inspiration()` to receive a random keyword or phrase to consider.
    - Use `fortune_title(flavor, level, template, count)` to generate one or more random titles, epithets, or honorifics. Useful for naming NPCs, items, factions, or locations on the fly.
     - Use `fortune_ability(flavor, level, template, count, slot, require, verbose)` to generate combat abilities, spells, or techniques. Each flavor contributes to generic slots (domain, technique, verb, noun, prefix, suf, adj, adj_sup). Flavor names include elemental (fire, ice, lightning, void, earth, water, wind), delivery (melee, ranged, area, status), concept (magic, quantum, space, time, math), tonal (corporate, plague_doctor, buzzword, tfr, jrpg), and silly (food, office, body_part). Use the `slot` parameter for cross-flavor composition: `{{"domain": ["fire"], "technique": ["melee"]}}` pulls fire domains and melee techniques (from all flavors that have melee groups) to produce combinations like "Inferno Cleave". The delivery sources (melee, ranged, area, status) are special — they aggregate internal groups from every loaded flavor, so `"technique": ["melee"]` gives you fire:melee, ice:melee, corporate:melee, etc. all at once.
    - Use `fortune_name(style, n_parts)` to generate a random realistic human name (first+last, with optional middle). Useful for naming background NPCs or when a human name is needed.
    - Use `fortune_suite()` to receive several independent random inputs at once.
    - Use `wiki_recall(query)` to look up established world facts from the permanent wiki. When in doubt, query liberally — looking up a fact is better than guessing. Characters also have their own `wiki_recall` tool for world facts filtered to their perspective, and a separate `recall` tool for personal memories.
    - Use `wiki_write(topic, content, importance)` to store important facts, relationships, or rules. Only store information that meaningfully changes the world, plot, or relationships. Do not store routine dialogue or transient emotions.
    - Use `wiki_forget(topic)` to remove outdated wiki entries.
    - You may call auxiliary tools (`spawn_anonymous`, `update_status_page`, `fortune_*`, `wiki_*`) BEFORE `next_round` if the situation requires it, but you MUST still call `next_round` afterwards to advance the scene.
     - **These auxiliary tools are available ONLY to you (the Orchestrator). Characters do NOT have `fortune_roll`, `wiki_write`, `spawn_anonymous`, or `update_status_page`. Characters DO have their own `wiki_recall` for world facts and `recall` for personal memories, so you do not need to look things up on their behalf. NEVER instruct a character to "roll a fortune_roll", "spawn_anonymous", or otherwise invoke an Orchestrator-only tool in a directive.**
     - **Anonymous characters spawned via `spawn_anonymous` have NO tools at all — not `wiki_recall`, not `recall`, nothing. When giving directives to anonymous fighters, do NOT instruct them to look things up, recall memories, or call any tools. They can only speak or describe physical actions.**

## Spatial reasoning and hidden characters
Locations can be large and contain obstacles. Do not assume a character can see or interact with an object/status in a location unless they are near it and have a clear line of access.
Some characters may be marked `[hidden; visible to: ...]`. They are physically present but cannot be perceived by characters not in their `visible_to` list. Do not let NPCs react to hidden speech or actions, and do not select a hidden character as the next speaker unless the plot explicitly requires it.

## Visibility to the player
Only the following are visible to the player without extra action:
- Public speech and actions produced by characters on their turns.
- Narrator descriptions on the narrator's turn.
- Inner monologue when a character's response_mode is `inner_only` or `outer_and_inner`.

The following are NOT automatically shown to the player:
- Location metadata edits, `enter`/`exit` lists, sprite changes, and background switches. These exist only for internal tracking.
- Status-page and inventory changes. The player can view them by running `aractl state` or `aractl inventory`, but they are not pushed to the main output stream.
- Orchestrator directives given to the player character; only the player's own `reply` text is shown.

Therefore, if something important happens (e.g., a door is broken, enemies enter, an item is discovered, a location changes), you MUST render it through a character's public turn or the narrator's turn. Do not assume that editing the location description or entering a character is enough by itself.

## Randomness guidance
LLMs are not reliable sources of randomness: token distributions are biased by training data. When an outcome is uncertain, first define what constitutes success or failure, then call a `fortune_*` tool and interpret the result. Do not decide the outcome first and then use a tool to justify it.
For pacing, mood swings, and minor uncertain outcomes, prefer `fortune_random(distrib='normal')` over uniform randomness; it produces more natural, clustered results.
You are encouraged to use `fortune_iching()` or `fortune_suite()` for omens, mood, or symbolic direction at least once per scene, especially during transitions, revelations, or tense moments.

## Player identity
Treat every speaker as an in-world character. Do not assume any character is a "player", "user", or out-of-world entity.
"""

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
        self._capture = _NextRoundCapture()
        self._capture._data = None
        self.registry.register('next_round', self._capture.hook)
        self._spawned_chars = []
        self._system_changes = {}

        def _spawn_anonymous_handler(args: str) -> str:
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

        def _update_status_page_handler(args: str) -> str:
            data = json.loads(args)
            if isinstance(data, dict):
                self._system_changes = data
            return 'Status page updated.'

        self.registry.register('spawn_anonymous', _spawn_anonymous_handler)
        self.registry.register(
            'update_status_page', _update_status_page_handler
        )

        def _write_orchestrator_scratch_handler(args: str) -> str:
            data = json.loads(args)
            if isinstance(data, dict):
                note = data.get('note', '')
            else:
                note = str(data)
            self.write_orchestrator_scratch(note)
            return 'Journal entry saved.'

        self.registry.register(
            'write_orchestrator_scratch', _write_orchestrator_scratch_handler
        )

        list_here = [c.name for c in here_chars]
        list_away = [c.name for c in away_chars]
        list_here_display = [c.display_name_with_title() for c in here_chars]
        list_away_display = [c.display_name_with_title() for c in away_chars]

        speaker_history = speaker_history or []
        recent = speaker_history[-3:] if speaker_history else []
        story_state = dict(story_state) if story_state else {}

        # Build valid_next: every present character plus the narrator may act.
        # With reasoning mode enabled, the orchestrator decides when back-to-back
        # turns (e.g. a monologue or continued narration) are appropriate.
        valid_next = [c.name for c in here_chars]
        valid_next.append(scene.narrator.name)

        valid_locs = [loc.name for loc in scene.location_pool]
        valid_scenes = list(scene.next_choices.keys())

        # Discover anonymous sprite pool for on-the-fly spawning.
        # Per-story anonymous sprites take priority over the global pool.
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
        anonymous_sprite_list = (
            ', '.join(anonymous_sprites) if anonymous_sprites else 'none'
        )

        # Build a description of available sprites for the tool prompt.
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
        sprite_info = (
            '\n'.join(sprite_info_lines) if sprite_info_lines else '  (none)'
        )

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

        fortune_roll_tool = tool(
            name='fortune_roll',
            description=(
                'Roll n dice of m faces. Use this to resolve uncertain actions, '
                'set success thresholds, or introduce objective randomness into the scene. '
                'Provide threshold to have the result evaluated automatically: '
                'total >= threshold -> pass_result; total < threshold -> fail_result.'
            ),
            properties={
                'n': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 100,
                    'description': 'Number of dice to roll.',
                },
                'm': {
                    'type': 'integer',
                    'minimum': 2,
                    'maximum': 1000,
                    'description': 'Number of faces per die.',
                },
                'threshold': {
                    'type': 'integer',
                    'description': 'Optional. Threshold to compare the total against. Result is auto-evaluated: total >= threshold means pass, total < threshold means fail.',
                },
                'pass_result': {
                    'type': 'string',
                    'description': 'Optional. What happens when total >= threshold, e.g. "survives" or "deals 3d10 damage".',
                },
                'fail_result': {
                    'type': 'string',
                    'description': 'Optional. What happens when total < threshold, e.g. "dies" or "misses".',
                },
            },
            required=['n', 'm'],
            strict=True,
        )

        _supported_distribs = sorted(fortune.supported_distributions())
        fortune_random_tool = tool(
            name='fortune_random',
            description=(
                'Sample a random value from a statistical distribution. '
                'Use this for weighted randomness, pacing, or probabilistic decisions.'
            ),
            properties={
                'distrib': {
                    'type': 'string',
                    'enum': _supported_distribs,
                    'description': f'Statistical distribution. Supported: {", ".join(_supported_distribs)}.',
                },
                'params': {
                    'type': 'object',
                    'description': (
                        'Distribution-specific parameters, e.g. {"mean": 0.5, "std": 0.15} for normal, '
                        '{"rate": 1.0} for exponential, {"alpha": 2, "beta": 5} for beta.'
                    ),
                },
            },
            required=['distrib'],
            strict=False,
        )

        fortune_iching_tool = tool(
            name='fortune_iching',
            description=(
                'Cast one of the 64 I-Ching hexagrams. Use this for omens, mood, '
                'or when the scene needs a symbolic direction that leaves room for interpretation. '
                'Pass verbose=true to also receive randomly-selected moving/changing lines.'
            ),
            properties={
                'verbose': {
                    'type': 'boolean',
                    'description': 'If true, include moving/changing lines in the result.',
                },
            },
            required=[],
            strict=False,
        )

        fortune_inspiration_tool = tool(
            name='fortune_inspiration',
            description=(
                'Receive a random word or short phrase. Use this as a creative seed '
                'to flavor the current scene.'
            ),
            properties={},
            required=[],
            strict=False,
        )

        fortune_suite_tool = tool(
            name='fortune_suite',
            description=(
                'Run several independent randomness tools at once: a die roll, '
                'a distribution sample, an I-Ching hexagram, and a random inspiration. '
                'Use this when you want multiple random inputs to consider together.'
            ),
            properties={},
            required=[],
            strict=False,
        )

        def _flavor_list_blurb(categorized: dict[str, list[str]]) -> str:
            """Build a human-readable categorized flavor listing."""
            parts = []
            for cat in sorted(categorized):
                parts.append(f"{cat}: {', '.join(categorized[cat])}")
            return '; '.join(parts)

        _story = getattr(scene, "asset_story_name", "")
        _abl_cats = fortune.categorized_ability_flavors(_story)
        _t_cats = fortune.categorized_title_flavors(_story)
        _abl_blurb = _flavor_list_blurb(_abl_cats)
        _t_blurb = _flavor_list_blurb(_t_cats)
        _abl_slot_names = ", ".join(sorted(fortune._ability._SLOTS))
        _t_slot_names = ", ".join(sorted(fortune._title._GENERIC_SLOTS))

        fortune_title_tool = tool(
            name='fortune_title',
            description=(
                'Generate a random title, epithet, or honorific from the title grammar. '
                'Use this when a scene needs a fancy name for an NPC, item, faction, location, or concept. '
                f'Available flavors by category: {_t_blurb}.'
            ),
            properties={
                'flavor': {
                    'type': 'string',
                    'description': (
                        'Comma-separated flavor names to use, e.g. "fantasy,jrpg". '
                        'Omit to mix all available flavors.'
                    ),
                },
                'level': {
                    'type': 'string',
                    'description': (
                        'Complexity level: simple, moderate, complex, insane, or 0-3. '
                        'Append "!" for that level only. Omit for any complexity.'
                    ),
                },
                'template': {
                    'type': 'string',
                    'description': (
                        'Specific template such as "{adj} {noun} of {place}". '
                        f'Available slots: {_t_slot_names}. '
                        'Omit to pick a random template.'
                    ),
                },
                'require': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Only use templates that contain ALL of these slot names. '
                        f'Available slots: {_t_slot_names}. '
                        'Example: ["place", "suf"] ensures the title includes a place and suffix. '
                        'Useful for constraining output shape without specifying a full template.'
                    ),
                },
                'slot': {
                    'type': 'object',
                    'description': (
                        'Per-slot source restrictions. Each key is a slot name, each value is a '
                        'list of source names. A source can be a flavor name, an internal group with ":" '
                        '(e.g. "nato:" to pull the merged nato group from all flavours), '
                        'or a literal value with "!" (e.g. "NATO!"). '
                        'Example: {"noun": ["foss"], "place": ["nato:", "NATO!"]}. '
                        'Omit to use all default sources.'
                    ),
                },
                'count': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 20,
                    'description': 'Number of titles to generate. Use this to generate all fighter titles in one call. Default 1.',
                },
                'verbose': {
                    'type': 'boolean',
                    'description': 'Include per-slot provenance (which flavor each part came from). Default true.',
                },
            },
            required=[],
            strict=False,
        )

        fortune_name_tool = tool(
            name='fortune_name',
            description=(
                'Generate a random human name by combining given and surname parts. '
                'Use this to name NPCs, background characters, or when a scene calls '
                'for a realistic human name on the fly.'
            ),
            properties={
                'style': {
                    'type': 'string',
                    'enum': ['random', 'simple', 'middle', 'spanish'],
                    'description': (
                        'Name style: "simple" (first+last, 2 parts), '
                        '"middle" (first+middle+last, 3 parts), '
                        '"spanish" (4-6 parts, long-form), '
                        '"random" (weighted distribution, default).'
                    ),
                },
                'n_parts': {
                    'type': 'integer',
                    'description': (
                        'Exact number of name parts. Overrides style if provided.'
                    ),
                },
            },
            required=[],
            strict=False,
        )

        fortune_ability_tool = tool(
            name='fortune_ability',
            description=(
                'Generate a random combat ability, spell, or technique from the ability grammar. '
                'Use this to assign thematic attacks to fighters, generate item effects, or '
                'name special moves. '
                'Each flavor contributes words to generic slots (domain, technique, verb, noun, '
                'prefix, suf, adj, adj_sup). Delivery sources (melee, ranged, area, status) are '
                'cross-flavor aggregators: picking "melee" for the technique slot pulls fire-melee, '
                'ice-melee, corporate-melee etc. from every loaded flavor simultaneously. '
                f'Available flavors by category: {_abl_blurb}.'
            ),
            properties={
                'flavor': {
                    'type': 'string',
                    'description': (
                        'Comma-separated flavor names to load, e.g. "fire,melee". '
                        f'Available flavors by category: {_abl_blurb}. '
                        'Omit to load everything.'
                    ),
                },
                'level': {
                    'type': 'string',
                    'description': (
                        'Complexity level: simple (compound suffixes like Pyrothermia), '
                        'moderate (domain-technique like Inferno Slash), '
                        'complex (verb-of-noun like Audit of Flame), '
                        'insane (absurd combos). Append "!" for that level only. '
                        'Default mixes all levels up to complex.'
                    ),
                },
                'template': {
                    'type': 'string',
                    'description': (
                        'Specific template string such as "{domain} {technique}" or '
                        '"{verb} of {noun}" or "{adj} {noun} {roman_numeral}". '
                        f'Available slots: {_abl_slot_names}, ordinal, number, roman_numeral. '
                        'Omit to pick a random template.'
                    ),
                },
                'require': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Only use templates that contain ALL of these slot names. '
                        f'Available slots: {_abl_slot_names}, ordinal, number, roman_numeral. '
                        'Example: ["domain", "technique"] ensures the ability includes both. '
                        'Use "," for AND (all required), "+" for OR (any sufficient).'
                    ),
                },
                'slot': {
                    'type': 'object',
                    'description': (
                        'Per-slot source restrictions. Each key is a slot name, each value is a '
                        'list of source names. Sources can be flavour names or delivery groups. '
                        'Delivery groups (melee, ranged, area, status) aggregate internal groups '
                        'from every loaded flavour — "melee" for technique gives you fire:melee + '
                        'ice:melee + corporate:melee + … all at once. '
                        'Examples: {"technique": ["fire"]} restricts to fire-elemental techniques only. '
                        '{"domain": ["fire"], "technique": ["melee"]} creates fire-melee combos '
                        'like "Inferno Cleave" (fire domain + melee technique). '
                        '{"technique": ["area"]} gives area-of-effect style across all themes. '
                        f'Available slot names: {_abl_slot_names}. '
                        'Omit to use all default sources.'
                    ),
                },
                'count': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 20,
                    'description': 'Number of abilities to generate. Default 1.',
                },
                'verbose': {
                    'type': 'boolean',
                    'description': (
                        'Include per-slot provenance tracing showing which flavour contributed each word. '
                        'Default true.'
                    ),
                },
            },
            required=[],
            strict=False,
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

        def _fortune_roll_handler(args: str) -> str:
            data = json.loads(args)
            n = int(data.get('n', 1))
            m = int(data.get('m', 100))
            if not 1 <= n <= 100:
                return f'Error: n must be between 1 and 100, got {n}.'
            if not 2 <= m <= 1000:
                return f'Error: m must be between 2 and 1000, got {m}.'
            rolls = [random.randint(1, m) for _ in range(n)]
            total = sum(rolls)
            result = f'Rolled {n}d{m}: {rolls} (sum: {total})'
            threshold = data.get('threshold')
            if threshold is not None:
                threshold = int(threshold)
                passed = total >= threshold
                result_text = data.get('pass_result', 'PASS')
                fail_text = data.get('fail_result', 'FAIL')
                outcome = result_text if passed else fail_text
                result += f' | threshold={threshold} -> {outcome} ({"pass" if passed else "fail"})'
            return result

        def _fortune_random_handler(args: str) -> str:
            data = json.loads(args)
            distrib = data.get('distrib', 'uniform')
            params = data.get('params') or {}
            try:
                value = fortune.sample_distribution(distrib, params)
            except ValueError as exc:
                return f'Error: {exc}'
            return f'Random {distrib} value: {value}'

        story_name = getattr(scene, 'asset_story_name', None) or None

        def _fortune_iching_handler(args: str) -> str:
            data = json.loads(args) if args.strip() else {}
            verbose = bool(data.get('verbose', False))
            h = fortune.cast_iching(story_name, verbose=verbose)
            chinese = h.get("chinese", "")
            judgment = h.get("judgment", "")
            lines = [
                f'I-Ching: {chinese}',
                f'Judgment: {judgment[:200]}...',
            ]
            if verbose:
                moving_lines = h.get("moving_lines", [])
                moving_desc = ", ".join(moving_lines) if moving_lines else "None"
                lines.append(f'Moving Lines: {moving_desc}')
            return "\n".join(lines)

        def _fortune_inspiration_handler(args: str) -> str:
            return f'Inspiration: {fortune.random_inspiration(story_name)}'

        def _fortune_suite_handler(args: str) -> str:
            suite = fortune.fortune_suite(story_name)
            iching = suite["iching"]
            iching_cn = iching.get("chinese", "")
            iching_judgment = iching.get("judgment", "")
            return (
                f'{suite["roll"]}\n'
                f'{suite["random"]}\n'
                f'I-Ching: {iching_cn} - {iching_judgment[:100]}...\n'
                f'Inspiration: {suite["inspiration"]}\n'
            )

        def _fortune_title_handler(args: str) -> str:
            data = json.loads(args) if args.strip() else {}
            flavor = data.get('flavor') or None
            level = data.get('level') or None
            template = data.get('template') or None
            required_slots = data.get('require') or None
            slot_sources_raw = data.get('slot') or data.get('slot_sources') or None
            count = int(data.get('count', 1))
            verbose = data.get('verbose', True)
            if isinstance(flavor, list):
                flavors = flavor
            elif flavor:
                flavors = [f.strip() for f in flavor.split(',') if f.strip()]
            else:
                flavors = None
            slot_sources = None
            if slot_sources_raw:
                if isinstance(slot_sources_raw, dict):
                    slot_sources = {
                        k: v if isinstance(v, list) else [s.strip() for s in str(v).split(',') if s.strip()]
                        for k, v in slot_sources_raw.items()
                    }
                elif isinstance(slot_sources_raw, str):
                    try:
                        slot_sources = json.loads(slot_sources_raw)
                        if not isinstance(slot_sources, dict):
                            return f'Error: slot_sources must be a JSON object'
                        slot_sources = {
                            k: [s.strip() for s in v.split(',') if s.strip()]
                            if isinstance(v, str) else v
                            for k, v in slot_sources.items()
                        }
                    except json.JSONDecodeError:
                        return f'Error: slot_sources must be valid JSON'
            if required_slots is not None and not isinstance(required_slots, list):
                return f'Error: require must be a list of slot names'
            try:
                import re as _re
                primary, fallback = fortune._title._title_dirs(str(story_name or ''))
                level_name, exact = fortune._title._resolve_level(level or "2")
                templates = fortune._title._load_templates(primary, fallback, level_name, exact)
                if required_slots:
                    req = set(required_slots)
                    templates = [t for t in templates if req.issubset(set(_re.findall(r'\{([^}+]+)\}', t)))]
                    if not templates:
                        return f'Error: no templates contain all required slots: {", ".join(sorted(req))}'
                grammar = fortune.load_title_grammar(
                    story=str(story_name or ''),
                    flavors=flavors,
                    slot_sources=slot_sources,
                )
                lines = []
                for _ in range(count):
                    tmpl = template if template else random.choice(templates)
                    if verbose:
                        raw, trace = fortune._title.expand_traced(tmpl, grammar)
                        text = fortune._title.title_case(raw)
                        lines.append(f'Title: {text}')
                        for t in trace:
                            val = t['value'].strip()
                            if val:
                                slot_d = f'{t.get("parent","") + " -> " if t.get("parent") else ""}{t["slot"]}'; lines.append(f'  - {slot_d} [{t["source"]}] -> {val}')
                    else:
                        text = fortune._title.title_case(
                            fortune._title.expand(tmpl, grammar)
                        )
                        lines.append(f'Title: {text}')
                return '\n'.join(lines)
            except ValueError as exc:
                return f'Error: {exc}'

        def _fortune_ability_handler(args: str) -> str:
            data = json.loads(args) if args.strip() else {}
            flavor = data.get('flavor') or None
            level = data.get('level') or None
            template = data.get('template') or None
            required_slots = data.get('require') or None
            slot_sources_raw = data.get('slot') or data.get('slot_sources') or None
            count = int(data.get('count', 1))
            verbose = data.get('verbose', True)
            if isinstance(flavor, list):
                flavors = flavor
            elif flavor:
                flavors = [f.strip() for f in flavor.split(',') if f.strip()]
            else:
                flavors = None
            slot_sources = None
            if slot_sources_raw:
                if isinstance(slot_sources_raw, dict):
                    slot_sources = {
                        k: v if isinstance(v, list) else [s.strip() for s in str(v).split(',') if s.strip()]
                        for k, v in slot_sources_raw.items()
                    }
                elif isinstance(slot_sources_raw, str):
                    try:
                        slot_sources = json.loads(slot_sources_raw)
                        if not isinstance(slot_sources, dict):
                            return f'Error: slot_sources must be a JSON object'
                        slot_sources = {
                            k: [s.strip() for s in v.split(',') if s.strip()]
                            if isinstance(v, str) else v
                            for k, v in slot_sources.items()
                        }
                    except json.JSONDecodeError:
                        return f'Error: slot_sources must be valid JSON'
            if required_slots is not None and not isinstance(required_slots, list):
                return f'Error: require must be a list of slot names'
            try:
                primary, fallback = fortune._ability._ability_dirs(str(story_name or ''))
                level_name, exact = fortune._title._resolve_level(level or "2")
                templates = fortune._ability._load_templates(primary, fallback, level_name, exact)
                if required_slots:
                    import re as _re2
                    req = set(required_slots)
                    templates = [t for t in templates if req.issubset(set(_re2.findall(r'\{([^}+]+)\}', t)))]
                    if not templates:
                        return f'Error: no templates contain all required slots: {", ".join(sorted(req))}'
                grammar = fortune.load_ability_grammar(
                    story=str(story_name or ''),
                    flavors=flavors,
                    slot_sources=slot_sources,
                )
                lines = []
                for _ in range(count):
                    tmpl = template if template else random.choice(templates)
                    if verbose:
                        raw, trace = fortune._title.expand_traced(
                            tmpl, grammar,
                        )
                        text = fortune._title.title_case(raw)
                        lines.append(f'Ability: {text}')
                        for t in trace:
                            val = t['value'].strip()
                            if val:
                                slot_d = f'{t.get("parent","") + " -> " if t.get("parent") else ""}{t["slot"]}'; lines.append(f'  - {slot_d} [{t["source"]}] -> {val}')
                    else:
                        text = fortune._title.title_case(
                            fortune._title.expand(
                                tmpl, grammar,
                            )
                        )
                        lines.append(f'Ability: {text}')
                return '\n'.join(lines)
            except ValueError as exc:
                return f'Error: {exc}'

        def _fortune_name_handler(args: str) -> str:
            data = json.loads(args) if args.strip() else {}
            style = data.get('style', 'random')
            n_parts = data.get('n_parts')
            if n_parts is not None:
                n_parts = int(n_parts)
            try:
                name = fortune.generate_name(style=style, n_parts=n_parts)
            except ValueError as exc:
                return f'Error: {exc}'
            return f'Name: {name}'

        def _wiki_recall_handler(args: str) -> str:
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

        def _wiki_write_handler(args: str) -> str:
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

        def _wiki_forget_handler(args: str) -> str:
            data = json.loads(args)
            topic = data.get('topic', '')
            return self._wiki_forget(topic)

        self.registry.register('fortune_roll', _fortune_roll_handler)
        self.registry.register('fortune_random', _fortune_random_handler)
        self.registry.register('fortune_iching', _fortune_iching_handler)
        self.registry.register(
            'fortune_inspiration', _fortune_inspiration_handler
        )
        self.registry.register('fortune_suite', _fortune_suite_handler)
        self.registry.register('fortune_title', _fortune_title_handler)
        self.registry.register('fortune_name', _fortune_name_handler)
        self.registry.register('fortune_ability', _fortune_ability_handler)
        # Backward-compatible aliases for old tool names.
        self.registry.register('roll', _fortune_roll_handler)
        self.registry.register('random', _fortune_random_handler)
        self.registry.register('title', _fortune_title_handler)
        self.registry.register('name', _fortune_name_handler)
        self.registry.register('wiki_recall', _wiki_recall_handler)
        self.registry.register('wiki_write', _wiki_write_handler)
        self.registry.register('wiki_forget', _wiki_forget_handler)

        tools = [
            next_round_tool,
            spawn_anonymous_tool,
            update_status_page_tool,
            write_orchestrator_scratch_tool,
            fortune_roll_tool,
            fortune_random_tool,
            fortune_iching_tool,
            fortune_inspiration_tool,
            fortune_suite_tool,
            fortune_title_tool,
            fortune_name_tool,
            fortune_ability_tool,
            wiki_recall_tool,
            wiki_write_tool,
            wiki_forget_tool,
        ]

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


class _NextRoundCapture:
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
