"""Story runner that manages scene transitions and persistent character state.

The :class:`Story` class is the "completed plot subsystem" missing from the
original proof-of-concept.  It loads scenes sequentially, carries character
scratchpad and memory state across scene boundaries, and finalises each scene
by running end-of-scene scratch updates for important characters.
"""

from __future__ import annotations

import concurrent.futures
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.context import ConversationContext
from ara.llm.tools import ToolRegistry, tool
from ara.memory.chroma import ChromaStore
from ara.memory.story_memory import StoryMemory
from ara.llm.models import GameRole
from ara.world.character import Importance, create_anonymous_character
from ara.utils.debug import DebugConsole
from ara.utils.logger import get_logger
from ara.memory.knowledge import CharacterMemory
from ara.world.character import Character
from ara.world.engine import Engine

from ara.world.registry import AssetRegistry
from ara.world.scene import Location, Scene
from ara.world.i18n import normalize_language
from ara.world.setting import WorldSetting, load_world_setting, resolve_world_setting_path
from ara.world.summarizer import Summarizer, SceneStateModifiers

logger = get_logger(__name__)


def _merge_characters(prev_scene: Scene, new_scene: Scene) -> None:
    """Carry over runtime state when a scene reload created fresh objects.

    :class:`Scene.load` is normally called with ``live_characters``, in which case
    the same object is reused and no merge is needed. This helper only copies
    state when the previous and new objects are genuinely different instances
    (e.g. fallback paths without a live cache).

    :param prev_scene: Scene that just ended.
    :param new_scene: Scene that is about to begin.
    """
    prev_by_canonical = {c.canonical_name: c for c in prev_scene.character_pool}
    carried = 0
    for char in new_scene.character_pool:
        old = prev_by_canonical.get(char.canonical_name)
        if old is None or old is char or char.importance < Importance.IMPORTANT:
            continue
        char.memory = old.memory
        char.scratch = old.scratch
        char.title = old.title
        char.status = dict(old.status)
        char.current_sprite = old.current_sprite
        char.hidden = old.hidden
        char.visible_to = set(old.visible_to)
        char.inner_log = list(old.inner_log)
        char.prev_scene_summary = old.prev_scene_summary
        carried += 1
    logger.debug(f"Carried over memory and scratch for {carried} characters into new scene")


def _finalize_character(
    char: Character,
    scene: Scene,
    engine: Engine,
    directives_log: dict[Character, str],
) -> None:
    """Run end-of-scene scratch update + memory summary for a single character."""
    scratch_tool = tool(
        name="write_scratch",
        description="""Update your scratchpad based on the conversation.
Use this space to reason and come up with plans.
Make guesses on when you might meet the other character(s) again.
Clean up and only keep what will be useful to carry over into future conversations.
If the scratchpad does not need changing, do not provide the `contents` field.""",
        properties={
            "contents": {
                "type": "string",
                "description": "New scratch content. Replaces all previous content.",
            }
        },
        required=[],
        strict=True,
    )

    registry = ToolRegistry()

    def _make_scratch_hook(target: Character):
        def scratch_hook(args: str) -> str:
            """Update *target*'s scratchpad from a tool-call argument."""
            data = json.loads(args)
            if "contents" in data:
                target.scratch.text = data["contents"]
                return "Updated scratch."
            return "No changes."
        return scratch_hook

    registry.register("write_scratch", _make_scratch_hook(char))

    prompt = (
        f"{char.name}, the current conversation has ended. "
        "Follow the instructions in your system prompt."
    )
    system = f"""IMPORTANT: Write scratch in {scene.language} only!
The current round of conversation has ended.
# Role
 - You are the ephemeral scratch-writing agent representing {char.name}.
 - Write how you think {char.name} would reply based on {char.name}'s previous messages.

## Instructions
 - Based on the previous rounds of conversation, update your scratchpad.
 - Use this space to reason and come up with plans.
 - Make guesses on when you might meet the other character(s) again.
 - Clean up and only keep what will be useful to carry over into future conversations.

## Additional directives given to {char.name}
    - {directives_log.get(char, "None")}
"""

    ctx = ConversationContext(char.name)
    ctx.concat_context(char.whoami)
    ctx.concat_context(char.scratch_context)
    # Include the scene conversation visible to this character so the
    # scratch-writer knows what actually happened. Use the character's curated
    # view so the scratch-writer remains the only assistant in its own history.
    ctx.concat_context(engine.ctx.curated_view(char.canonical_name))
    ctx.user_message(prompt, name="System")

    result = engine.client.complete(
        role=GameRole.CHARACTER,
        system_prompt=system,
        messages=ctx.to_list(),
        tools=[scratch_tool],
        tool_choice="auto",
        stream=False,
        name=char.name,
    )

    for tc in result.tool_calls:
        registry.call(tc["function"]["name"], tc["function"]["arguments"])

    # Generate a structured memory summary via sub-agent
    summary = engine.client.complete_subagent(
        task=f"Summarise what {char.name} experienced and learned in this scene. "
             f"Focus on facts, relationships, and open questions. "
             f"Write from {char.name}'s perspective in {scene.language}.",
        context=char.scratch.text,
        max_tokens=256,
    )
    if summary.strip():
        char.memory.add_conversation([summary.strip()])
        logger.debug(f"Stored memory summary for {char.name}")


def _finalize_orchestrator(
    scene: Scene,
    engine: Engine,
    directives_log: dict[Character, str],
) -> None:
    """Run an end-of-scene journal update for the orchestrator.

    The orchestrator scratch is treated as a cross-scene diary.  At the end of
    each scene it is replaced with a distilled scene-level summary that the next
    scene's orchestrator can read at a glance.
    """
    transcript = ConversationContext.to_narrative_text(
        engine.ctx.curated_view("__orchestrator__", collapse=False),
        observer_name="Orchestrator",
    )

    scratchpads = {
        c.name: c.scratch.text
        for c in scene.character_pool
        if c.scratch.text and c.scratch.text != "Nothing yet!"
    }
    orch_scratch = engine.orchestrator.scratch.text
    if orch_scratch and orch_scratch != "Nothing yet!":
        scratchpads["Orchestrator"] = orch_scratch

    scratch_section = "\n\n".join(
        f"--- {name}'s scratchpad ---\n{text}"
        for name, text in scratchpads.items()
    ) or "(No scratchpads available.)"

    summary = engine.client.complete_subagent(
        task=f"You are the director of the scene that just ended. Distill it into a concise journal entry for the next scene's orchestrator. Focus on established facts, unresolved threads, and anything the next orchestrator should keep in mind. Write in {scene.language} only.",
        context=f"Scene transcript:\n{transcript}\n\nPrivate scratchpads:\n{scratch_section}",
        max_tokens=512,
    )
    if summary.strip():
        engine.orchestrator.scratch.text = summary.strip()
        logger.debug("Updated orchestrator scratch after scene")


def _finalize_scene(
    scene: Scene,
    engine: Engine,
    directives_log: dict[Character, str],
) -> None:
    """Run end-of-scene scratch updates for important characters.

    Only characters with :attr:`Importance.IMPORTANT` or higher participate.
    The player and narrator are skipped.

    Per-character work runs in parallel with bounded concurrency.

    :param scene: The scene that just ended.
    :param engine: Active engine instance (provides the LLM client).
    :param directives_log: Mapping from character to the last directive they
        received during the scene.
    """
    targets = [
        char for char in scene.character_pool
        if char not in (scene.player, scene.narrator)
        and char.importance >= Importance.IMPORTANT
    ]

    # Finalize the orchestrator's journal.  This runs regardless of whether
    # there are character targets so the director always gets a scene-level
    # summary before the next scene loads.
    _finalize_orchestrator(scene, engine, directives_log)

    if not targets:
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_finalize_character, char, scene, engine, directives_log)
            for char in targets
        ]
        for future in concurrent.futures.as_completed(futures):
            exc = future.exception()
            if exc is not None:
                logger.warning(f"Finalization failed for a character: {exc}")


def _archive_scratchpads(
    scene: Scene,
    engine: Engine,
) -> None:
    """Archive current scratchpads into ``prev_text`` after summarization.

    This is called by :meth:`Story._run_finalization` after the summarizer has
    had a chance to read the final in-scene scratchpads.
    """
    for char in scene.character_pool:
        if char not in (scene.player, scene.narrator) and char.importance >= Importance.IMPORTANT:
            char.scratch.prepare_for_new_scene()
    engine.orchestrator.scratch.prepare_for_new_scene()


@dataclass
class StoryStep:
    """Result of a single :meth:`Story.step` call.

    :ivar event: What happened this tick.  One of ``scene_loaded``,
        ``turn``, ``needs_player_input``, ``transition``, ``story_complete``.
    :ivar phase: Phase of a ``transition`` event (currently ``ended``).
    :ivar scene: The newly-loaded scene (only for ``scene_loaded``).
    :ivar suggestions: Orchestrator suggestions (only for ``needs_player_input``).
    :ivar next_scene: Identifier of the next scene (only for ``transition``).
    :ivar next_scene_name: Human-readable name of the next scene, taken from
        the scene TOML (only for ``transition``).
    :ivar loading_background: Filename stem for the loading-screen image
        (only for ``transition``).
    :ivar speaker: Name of the character whose turn this was.
    :ivar enter: Names of characters that entered this turn.
    :ivar exit: Names of characters that exited this turn.
    :ivar spawn: Names of anonymous characters spawned this turn.
    :ivar sprite_changes: Mapping of character names → new sprite names.
    :ivar switch_background: Background stem activated for the current location.
    :ivar location: Current location (for scene_loaded/turn/needs_player_input).
    :ivar output: Text produced by this tick (spoken dialogue or narration).
        Empty for non-speech ticks such as scene transitions.
    :ivar system_changes: Updates to the player system page applied this turn.
    """

    event: str
    phase: str | None = None
    scene: Scene | None = None
    suggestions: list[str] | None = None
    next_scene: str | None = None
    next_scene_name: str | None = None
    loading_background: str | None = None
    speaker: str | None = None
    speaker_title: str = ""
    enter: list[str] = field(default_factory=list)
    exit: list[str] = field(default_factory=list)
    spawn: list[str] = field(default_factory=list)
    sprite_changes: dict[str, str] = field(default_factory=dict)
    switch_background: str = ""
    system_changes: dict[str, Any] = field(default_factory=dict)
    location: Location | None = None
    output: str = ""
    inner: str = ""


class Story:
    """Top-level controller that runs a sequence of scenes.

    A :class:`Story` instance is initialised with a starting scene file.
    Calling :meth:`start` followed by repeated :meth:`step` drives the story
    as a state machine:

    1. Load the next scene TOML.
    2. Merge character state from the previous scene.
    3. Run the :class:`Engine` one turn at a time.
    4. Finalise the scene (end-of-scene scratch updates).
    5. Resolve the chosen next scene and repeat.

    The old blocking :meth:`run` convenience wrapper is still available.

    :param config: Application settings.
    :param db: Shared ChromaDB store.
    :param client: LLM client instance.
    :param initial_scene_path: Path to the first scene ``.toml`` file.
    """

    def __init__(
        self,
        config: AraSettings,
        db: ChromaStore,
        client: LLMClient,
        initial_scene_path: Path,
    ) -> None:
        """Create a story controller.

        :param config: Application settings.
        :param db: Shared ChromaDB store.
        :param client: LLM client instance.
        :param initial_scene_path: Path to the first scene ``.toml`` file.
        """
        self.config = config
        self.db = db
        self.client = client
        self.engine = Engine(client, db=db)
        self.initial_scene_path = initial_scene_path
        self._scene_history: list[str] = []
        self._scene_names: dict[str, str] = {}
        self._current_path: Path | None = None
        self._prev_scene: Scene | None = None
        self._current_scene: Scene | None = None
        self._state: str = "idle"
        self._skipped_scene: bool = False
        self._summarizer = Summarizer(client)
        self._next_scene_summaries: dict[str, str] = {}
        self._next_scene_location_desc: str = ""
        self._next_scene_location_descs: dict[str, str] = {}
        self._next_scene_time: str = ""
        self._finalize_turn_text: str = ""
        self._finalize_turn_changes: dict[str, Any] = {}
        self._next_scene_facts: list[dict[str, Any]] = []
        self._next_scene_player_status: dict[str, Any] = {}
        self._next_scene_free_status: dict[str, Any] = {}
        self._next_scene_location_statuses: dict[str, Any] = {}
        self._next_scene_state_modifiers = SceneStateModifiers()
        self._next_scene_character_status_updates: dict[str, dict[str, Any]] = {}
        self._next_scene_character_overrides: dict[str, dict[str, str]] = {}
        self._next_scene_anonymous_chars: dict[str, dict[str, str]] = {}
        self._next_scene_orchestrator_note: str = ""
        self._next_scene_wiki_context: str = ""
        self._character_status: dict[str, dict[str, Any]] = {}
        self._narrative_state: dict[str, Any] = {}
        self._story_meta: dict[str, str] = {}
        self._story_dir: Path = initial_scene_path.parent
        self._first_scene: str | None = None
        self._clear_char_memory_on_load: bool = False
        self._world_id: str = ""
        self._language: str = normalize_language(config.language)
        self._loaded_settings: set[str] = set()
        self.registry = AssetRegistry(config, db)
        self._story_memory = StoryMemory(db)
        self._live_characters: dict[str, Character] = {}
        self._live_locations: dict[str, Location] = {}
        self._archived_scene_snapshots: list[dict[str, Any]] = []
        self._registry_indexed: bool = False
        if initial_scene_path.name == "ini_scene.toml":
            self._story_dir = initial_scene_path.parent
            try:
                with initial_scene_path.open("rb") as f:
                    ini = tomllib.load(f)
                self._story_meta = {
                    k: str(v)
                    for k, v in ini.items()
                    if k not in ("first_scene", "world", "language")
                }
                self._first_scene = ini.get("first_scene")
                self._world_id = ini.get("world", "")
                if "language" in ini:
                    self._language = normalize_language(ini["language"])
            except Exception as exc:
                logger.warning(f"Failed to read ini_scene.toml: {exc}")

    @property
    def finished(self) -> bool:
        """``True`` when the story has no more scenes to play."""
        return self._state == "complete"

    @property
    def current_scene(self) -> Scene | None:
        """The scene currently being played, or ``None``."""
        return self._current_scene

    @property
    def scene_history(self) -> list[str]:
        """List of scene identifiers visited so far."""
        return self._scene_history

    @property
    def language(self) -> str:
        """Current story language. Updated when a scene TOML declares a language."""
        return self._language

    def _ensure_registry(self) -> None:
        """Index the story's characters and locations if not already done."""
        if self._registry_indexed:
            return
        self.registry.index_story(self._story_dir, self._language)
        self._registry_indexed = True

    def start(self, scene_id: str | None = None, clear_history: bool = False) -> None:
        """Prepare the story for playback.

        Must be called before the first :meth:`step`.

        :param scene_id: Optional scene identifier to start at instead of the
            initial scene path.  The identifier is resolved relative to the
            story directory.
        :param clear_history: If ``True``, erase all persisted ChromaDB
            collections so that a fresh start does not inherit summaries,
            wiki entries, or character memories from previous runs.
        """
        self._ensure_registry()
        if scene_id:
            self._current_path = self._resolve_scene_path(scene_id)
        elif self._first_scene:
            self._current_path = self._resolve_scene_path(self._first_scene)
        else:
            self._current_path = self.initial_scene_path
        self._prev_scene = None
        self._scene_history = []
        self._archived_scene_snapshots = []
        self._current_scene = None
        self._state = "loading"
        self._loaded_settings = set()

        self._clear_char_memory_on_load = clear_history

        if clear_history and self.db is not None:
            self.db.clear_all_collections()

    def step(self) -> StoryStep:
        """Advance the story by one tick.

        A tick is one of: load a scene, run one engine turn, request player
        input, emit a transition marker, finalise a scene, or complete the story.

        :return: Description of what happened this tick.
        :raises RuntimeError: If the story is not started, is already complete,
            or is waiting for player input.
        """
        if self._state == "idle":
            raise RuntimeError("Story not started. Call start() first.")
        if self._state == "loading":
            return self._load_scene()
        if self._state == "finalizing":
            return self._run_finalization()
        if self._state == "running":
            if self._finalize_turn_text:
                text = self._finalize_turn_text
                self._finalize_turn_text = ""
                return StoryStep(
                    event="finalize_turn",
                    speaker=self.current_scene.narrator.name if self.current_scene else None,
                    output=text,
                    location=self.engine.loc,
                )
            if self.engine.needs_player_input:
                raise RuntimeError(
                    "Story is waiting for player input. Call submit_player_input() first."
                )
            result = self.engine.step()
            if result.needs_player_input:
                return StoryStep(
                    event="needs_player_input", suggestions=result.suggestions,
                    speaker=result.speaker,
                    speaker_title=result.speaker_title,
                    enter=result.enter,
                    exit=result.exit,
                    spawn=result.spawn,
                    sprite_changes=result.sprite_changes,
                    switch_background=result.switch_background,
                    system_changes=result.system_changes,
                    location=self.engine.loc,
                    output=result.output,
                )
            if result.scene_ended:
                return self._finalize_and_transition()
            return StoryStep(
                event="turn", speaker=result.speaker,
                speaker_title=result.speaker_title,
                enter=result.enter,
                exit=result.exit,
                spawn=result.spawn,
                sprite_changes=result.sprite_changes,
                switch_background=result.switch_background,
                system_changes=result.system_changes,
                location=self.engine.loc,
                output=result.output,
                inner=result.inner,
            )
        if self._state == "complete":
            raise RuntimeError("Story is already complete")
        raise RuntimeError(f"Unknown story state: {self._state}")

    def submit_attempt(self, attempt: str | dict[str, Any]) -> None:
        """Store a player action attempt without ending the turn."""
        self.engine.submit_attempt(attempt)

    def submit_player_input(
        self,
        text: str,
        attempt: str | dict[str, Any] | None = None,
    ) -> None:
        """Forward player input to the engine.

        Must only be called after :meth:`step` returned
        ``event="needs_player_input"``.
        """
        self.engine.submit_player_input(text, attempt=attempt)

    def generate_player_input(self, suggestion: str) -> str:
        """Generate natural player dialogue from a suggestion.

        :param suggestion: Suggestion text to expand.
        :return: Generated player dialogue.
        """
        return self.engine.generate_player_input(suggestion)

    def _load_world_setting_ids(
        self,
        world_id: str | None,
        settings: list[str] | None = None,
        story_name: str | None = None,
    ) -> None:
        """Load a list of world-setting TOMLs once per story run.

        The base ``world`` and any entries in ``settings`` are treated the same:
        they are world-setting files. Each distinct file is upserted into the
        wiki once, regardless of how many scenes prescribe it.
        """
        prescribed: list[str] = []
        if world_id:
            prescribed.append(world_id)
        for setting_id in settings or []:
            if setting_id not in prescribed:
                prescribed.append(setting_id)

        story_name = story_name or self._story_dir.name
        for setting_id in prescribed:
            if setting_id in self._loaded_settings:
                continue
            path = resolve_world_setting_path(setting_id, self.config, story_name)
            self._upsert_setting_file(path, label="world setting")
            self._loaded_settings.add(setting_id)

    def _load_world_settings(self, scene: Scene) -> None:
        """Load all world-setting TOMLs prescribed by *scene*."""
        story_name = scene.asset_story_name or self._story_dir.name
        self._load_world_setting_ids(
            scene.world or self._world_id, scene.settings, story_name=story_name
        )

    def _wiki_has_content(self) -> bool:
        """Return ``True`` if the orchestrator wiki collection has documents."""
        if self.db is None:
            return False
        try:
            return self.db.collection("orchestrator_wiki").count() > 0
        except Exception as exc:
            logger.debug(f"Could not check orchestrator_wiki content: {exc}")
            return False

    def _query_characters(self, query: str) -> list[str]:
        """Search available character cards for names matching *query*.

        Returns a short list of "Name: summary" lines. This is the summarizer's
        optional "expand roster" tool; it avoids dumping the whole cast into context.
        """
        query_lower = query.lower()
        results: list[str] = []
        assets_cc = self.config.characters_path(self._story_dir.name)
        if not assets_cc.exists():
            return results
        for char_dir in assets_cc.iterdir():
            if not char_dir.is_dir():
                continue
            name = char_dir.name
            if query_lower in name.lower():
                results.append(f"{name}: (name match)")
                continue
            card_path = char_dir / "card.toml"
            if card_path.exists():
                try:
                    with card_path.open("rb") as f:
                        card = tomllib.load(f)
                    text = " ".join(
                        str(card.get(k, "")) for k in ("summary", "personality", "scenario")
                    ).lower()
                    if query_lower in text:
                        summary = card.get("summary", "")
                        results.append(f"{name}: {summary}")
                except Exception:
                    continue
        return results[:10]

    def _upsert_setting_file(self, path: Path | None, label: str = "setting") -> None:
        """Load a single world-setting TOML and upsert its entries into the wiki."""
        if not path or not path.exists():
            logger.debug(f"No {label} found at {path}; skipping wiki upsert.")
            return

        try:
            setting = load_world_setting(path)
        except Exception as exc:
            logger.warning(f"Failed to load {label} from {path}: {exc}")
            return

        if self.db is None:
            return

        entries = setting.wiki_entries()
        if not entries:
            logger.debug(f"{label} '{setting.id}' has no wiki entries; skipping upsert.")
            return

        ids = list(entries.keys())
        docs = [entries[i] for i in ids]
        metadatas = [
            {"topic": topic, "importance": "critical", "world": setting.id}
            for topic in ids
        ]
        try:
            self.db.upsert("orchestrator_wiki", ids=ids, documents=docs, metadatas=metadatas)
            logger.info(f"Loaded {label} '{setting.id}' with {len(ids)} wiki entries")
        except Exception as exc:
            logger.warning(f"Failed to upsert {label} into wiki: {exc}")

    def _load_scene(self) -> StoryStep:
        """Load the scene at :attr:`_current_path` and start the engine."""
        self._ensure_registry()
        if self._current_path is None or not self._current_path.exists():
            self._state = "complete"
            return StoryStep(event="story_complete")

        scene = Scene.load(
            self._current_path,
            self.db,
            self.config,
            scene_history=self._scene_history,
            registry=self.registry,
            live_characters=self._live_characters,
            live_locations=self._live_locations,
        )
        # Sync story language with the scene. If the scene TOML explicitly
        # declares a language, adopt it as the new current story language.
        # Otherwise, ensure the scene uses the current story language.
        try:
            with self._current_path.open("rb") as f:
                raw_data = tomllib.load(f)
        except Exception:
            raw_data = {}
        if "language" in raw_data:
            self._language = normalize_language(raw_data["language"])
        scene.language = self._language
        self._current_scene = scene
        self._scene_names[scene.id] = scene.name
        self._scene_history.append(scene.id)

        # Clear character memory collections on fresh starts
        if self._clear_char_memory_on_load and self.db is not None:
            for char in scene.character_pool:
                if isinstance(char.memory, CharacterMemory):
                    try:
                        coll = self.db.collection(char.memory.collection_name)
                        existing = coll.get()
                        ids = existing.get("ids", []) or []
                        if ids:
                            coll.delete(ids=ids)
                    except Exception:
                        pass
            self._clear_char_memory_on_load = False

        # Fin scenes immediately end the story without running turns.
        if scene.scene_type == "fin":
            self._state = "complete"
            return StoryStep(event="story_complete", scene=scene)

        if self._prev_scene is not None:
            _merge_characters(self._prev_scene, scene)

        # Instantiate any anonymous NPCs explicitly declared by the transition
        # summarizer. They are added to the scene as starting characters.
        if self._next_scene_anonymous_chars:
            for name, data in self._next_scene_anonymous_chars.items():
                if scene.character_by_name(name) is None:
                    anon = create_anonymous_character(
                        name=name,
                        description=data.get("description", ""),
                        sprite=data.get("sprite", ""),
                    )
                    scene.character_pool.add(anon)
                    scene.starting_characters.add(anon)
            self._next_scene_anonymous_chars = {}

        # Apply any stored character status flags to the newly-loaded scene cast.
        for char in scene.character_pool:
            if char.canonical_name in self._character_status:
                char.status = dict(self._character_status[char.canonical_name])

        # Normalize and apply any character status updates emitted by the
        # summarizer for the next scene.
        for name, status in self._next_scene_character_status_updates.items():
            char = scene.character_by_name(name)
            if char is not None:
                self._character_status[char.canonical_name] = status
                char.status = dict(status)
        self._next_scene_character_status_updates = {}

        self._state = "running"
        previous_world_time = self.engine.world_time
        self.engine.start(scene)

        # Restore status pages carried over from the previous scene.
        if self._next_scene_player_status:
            self.engine._player_status = dict(self._next_scene_player_status)
            self._next_scene_player_status = {}
        if self._next_scene_free_status:
            self.engine._free_status = dict(self._next_scene_free_status)
            self._next_scene_free_status = {}
        if self._next_scene_location_statuses:
            for loc in scene.location_pool:
                if loc.canonical_name in self._next_scene_location_statuses:
                    loc.status = dict(self._next_scene_location_statuses[loc.canonical_name])
            self._next_scene_location_statuses = {}

        # Apply state modifiers from the transition summarizer. Normalize keys
        # from display names to canonical IDs using the newly-loaded scene.
        normalized_modifiers = self._normalize_state_modifiers(scene, self._next_scene_state_modifiers)
        self._apply_state_modifiers(scene, normalized_modifiers)
        self._next_scene_state_modifiers = SceneStateModifiers()

        # For the very first scene, ask the summarizer for initial state modifiers
        # after world settings have been loaded. Skip if there is nothing to modify.
        if self._prev_scene is None and (scene.character_pool or scene.location_pool):
            initial_modifiers = self._summarizer.apply_initial_state_modifiers(
                scene, language=self._language
            )
            self._apply_state_modifiers(scene, initial_modifiers)

        # Pass narrative state to the engine/orchestrator.
        self.engine._story_state = dict(self._narrative_state)

        # Load all world settings prescribed by the scene.
        self._load_world_settings(scene)

        # Inject summarizer-prefetched wiki context and orchestrator note.
        self.engine.orchestrator.prefetched_wiki = self._next_scene_wiki_context
        self.engine.orchestrator.orchestrator_note = self._next_scene_orchestrator_note
        self._next_scene_wiki_context = ""
        self._next_scene_orchestrator_note = ""

        # For the very first scene, run the same wiki prefetch now that world
        # settings have been upserted. Other finalization tasks are skipped.
        if self._prev_scene is None and self._wiki_has_content():
            self.engine.orchestrator.prefetched_wiki = self._summarizer.prefetch_wiki_context(
                plot=scene.plot_story,
                considerations=scene.plot_considerations or "",
                world=scene.world,
                zeitgeist=scene.zeitgeist,
                tone=scene.tone,
                language=self._language,
                wiki_recall_fn=self.engine.orchestrator.prefetch_wiki,
            )

        # Distribute per-character bridging summaries from the summarizer.
        # Keys may be display or canonical names; resolve against the new scene.
        if self._next_scene_summaries:
            for name, summary in self._next_scene_summaries.items():
                char = scene.character_by_name(name)
                if char is not None and summary:
                    char.prev_scene_summary = summary
            self._next_scene_summaries = {}

        # Apply per-character card-field overrides from the summarizer.
        if self._next_scene_character_overrides:
            for name, overrides in self._next_scene_character_overrides.items():
                char = scene.character_by_name(name)
                if char is not None and overrides:
                    char.card_overrides = dict(overrides)
            self._next_scene_character_overrides = {}

        # Apply finalized location descriptions and time from the summarizer.
        if self._next_scene_location_descs:
            for name, desc in self._next_scene_location_descs.items():
                loc = scene.location_by_name(name)
                if loc is not None:
                    loc.desc = desc
            self._next_scene_location_descs = {}
            self._next_scene_location_desc = ""
        elif self._next_scene_location_desc:
            scene.starting_location.desc = self._next_scene_location_desc
            self._next_scene_location_desc = ""
        if self._next_scene_time:
            scene.time = self._next_scene_time
            self._next_scene_time = ""
        elif not scene.time:
            scene.time = previous_world_time

        if self._skipped_scene and self.engine.ctx is not None:
            self.engine.ctx.user_message(
                "[SYSTEM NOTICE: A scene skip occurred. The narrative jumped directly to this scene. "
                "Characters may need to re-establish context.]",
                name="System",
            )
            self._skipped_scene = False
        return StoryStep(event="scene_loaded", scene=scene, location=scene.starting_location)

    def _finalize_and_transition(self) -> StoryStep:
        """Emit the end-of-scene transition marker and prepare to finalise."""
        assert self._current_scene is not None
        next_scene = self.engine.next_scene
        if next_scene and next_scene in self._current_scene.next_choices:
            next_choice_obj = self._current_scene.next_choices[next_scene]
            self._current_path = self._resolve_scene_path(next_choice_obj.id)
            self._state = "finalizing"
            loading_bg = self.engine.loc.loading_background if self.engine.loc else None
            next_scene_name = self._scene_names.get(next_scene)
            if next_scene_name is None and self._current_path.exists():
                try:
                    with self._current_path.open("rb") as f:
                        next_scene_name = tomllib.load(f).get("name", next_scene)
                except Exception:
                    next_scene_name = next_scene
            return StoryStep(
                event="transition",
                phase="ended",
                next_scene=next_scene,
                next_scene_name=next_scene_name or next_scene,
                loading_background=loading_bg or None,
            )

        self._current_path = None
        self._state = "complete"
        return StoryStep(event="story_complete")

    def _run_finalization(self) -> StoryStep:
        """Run end-of-scene finalisation and load the next scene."""
        assert self._current_scene is not None
        _finalize_scene(self._current_scene, self.engine, self.engine.directives_log)
        self._prev_scene = self._current_scene

        # Store scene summary in story-wide history
        scene = self._current_scene
        summary = (
            f"Scene: {scene.id}\n"
            f"Characters: {[c.name for c in scene.starting_characters]}\n"
            f"Plot: {scene.plot_story[:500]}\n"
            f"Next scene: {self.engine.next_scene or '(end)'}"
        )
        self.db.upsert(
            "story_history",
            ids=[f"scene_{scene.id}"],
            documents=[summary],
            metadatas=[{"scene_id": scene.id}],
        )

        # Run the transition summarizer to bridge context.
        next_scene = self.engine.next_scene
        if next_scene:
            self._run_summarizer(next_scene)
            self._generate_finalize_turn(next_scene)

        # Archive scratchpads now that finalization and summarization are done.
        _archive_scratchpads(self._current_scene, self.engine)

        # Telescope: archive a full snapshot of the scene that just ended.
        # This is inert gameplay baggage; it lets saves carry prior scene state.
        from ara.persistence.save import SaveManager
        manager = SaveManager(self.config)
        self._archived_scene_snapshots.append(
            manager._build_snapshot(self, queue=[])
        )

        self._state = "loading"
        return self._load_scene()

    def _run_summarizer(self, next_scene_id: str) -> None:
        """Generate a bridging summary and finalize location edits."""
        assert self._current_scene is not None
        assert self.engine.ctx is not None
        assert self.engine.loc is not None

        next_path = self._current_path
        if next_path is None or not next_path.exists():
            return

        try:
            import tomllib
            with next_path.open("rb") as f:
                next_data = tomllib.load(f)
        except Exception:
            return

        next_plot = next_data.get("plot", {}).get("scene", "")
        next_considerations = next_data.get("plot", {}).get("considerations", "")
        current_considerations = self._current_scene.plot_considerations or ""
        summarizer_considerations = self._current_scene.next_choices.get(
            next_scene_id
        )
        summarizer_considerations = (
            summarizer_considerations.summarizer_considerations
            if summarizer_considerations
            else ""
        )

        # Load the next scene's world settings early so the summarizer can search
        # them while prefetching wiki context.
        self._load_world_setting_ids(
            next_data.get("world"),
            next_data.get("settings", []),
            story_name=self._story_dir.name,
        )

        # Gather scratchpads from characters in the current scene, plus the orchestrator's.
        scratchpads = {
            c.name: c.scratch.text
            for c in self._current_scene.character_pool
            if c.scratch.text and c.scratch.text != "Nothing yet!"
        }
        orch_scratch = self.engine.orchestrator.scratch.text
        if orch_scratch and orch_scratch != "Nothing yet!":
            scratchpads["Orchestrator"] = orch_scratch

        # Recall relevant past-scene summaries from long-term memory.
        history_queries = [
            q for q in (next_plot, next_considerations, current_considerations) if q
        ]
        history_hits = self._story_memory.recall(history_queries, n_results=3)
        history_context = "\n\n".join(history_hits) if history_hits else ""

        # Peek at next scene's character pool from TOML and resolve display names.
        next_scene_canonicals: list[str] = []
        try:
            next_chars = next_data.get("character", {}).get("pool", [])
            if next_chars:
                next_scene_canonicals = list(next_chars)
        except Exception:
            pass

        def _display_name(canonical: str) -> str:
            meta = self.registry.get_character(canonical, self._language)
            return meta.get("display_name", canonical) if meta else canonical

        next_scene_chars = [_display_name(c) for c in next_scene_canonicals]
        next_player_canonical = next_data.get("character", {}).get("player", "")
        next_narrator_canonical = next_data.get("character", {}).get("narrator", "")
        next_player_name = _display_name(next_player_canonical) if next_player_canonical else ""
        next_narrator_name = _display_name(next_narrator_canonical) if next_narrator_canonical else ""

        location_descs = {loc.name: loc.desc for loc in self._current_scene.location_pool}
        next_scene_locations = next_data.get("location", {}).get("pool", [])

        # Build the next-scene cast. This is the authoritative list of characters
        # the summarizer must account for. It contains the next scene's named
        # pool (minus player/narrator, which are listed separately) plus any
        # anonymous NPCs currently present at the active location.
        # Characters that have already exited/left (away) must NOT be carried
        # forward automatically; otherwise anonymous extras from a previous
        # location leak into the next scene.
        here_chars = self.engine.here_chars
        current_char_names = {c.name for c in here_chars}
        anonymous_names = {
            c.name for c in here_chars
            if c.importance == Importance.ANONYMOUS
        }
        excluded = {next_player_name, next_narrator_name}
        next_scene_cast = [n for n in next_scene_chars if n not in excluded]
        plot_text = f"{next_plot}\n{next_considerations}"
        for name in current_char_names:
            if name in plot_text and name not in next_scene_cast and name not in anonymous_names and name not in excluded:
                next_scene_cast.append(name)
        # Anonymous characters currently in the scene are part of the cast so
        # the summarizer can preserve them explicitly instead of guessing.
        for name in anonymous_names:
            if name not in next_scene_cast:
                next_scene_cast.append(name)

        previous_scene_characters = [
            f"{name} [anonymous]" if name in anonymous_names else name
            for name in sorted(current_char_names)
        ]

        wiki_context = ""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            summary_future = executor.submit(
                self._summarizer.summarize_transition,
                current_scene=self._current_scene,
                current_scene_considerations=current_considerations,
                next_scene_plot=next_plot,
                next_scene_considerations=next_considerations,
                conversation_context=self.engine.ctx.curated_view(
                    "__orchestrator__", collapse=False
                ),
                location_desc=self.engine.loc.desc,
                language=self._language,
                scratchpads=scratchpads,
                next_scene_chars=next_scene_cast,
                next_player_name=next_player_name,
                next_narrator_name=next_narrator_name,
                previous_scene_characters=previous_scene_characters,
                location_descs=location_descs,
                next_scene_locations=next_scene_locations,
                mechanical_changelog=self.engine.mechanical_changelog,
                player_status=self.engine.player_status,
                world_time=self.engine.world_time,
                query_characters_fn=self._query_characters,
                next_scene_cast=next_scene_cast,
                current_character_status=self._character_status,
                narrative_state=self._narrative_state,
                summarizer_considerations=summarizer_considerations,
                history_context=history_context,
            )
            wiki_future = None
            if self._wiki_has_content():
                wiki_future = executor.submit(
                    self._summarizer.prefetch_wiki_context,
                    plot=next_plot,
                    considerations=next_considerations,
                    world=self._current_scene.world,
                    zeitgeist=self._current_scene.zeitgeist,
                    tone=self._current_scene.tone,
                    language=self._language,
                    wiki_recall_fn=self.engine.orchestrator.prefetch_wiki,
                )
            bridging_summaries, finalized_descs, finalized_time, facts, player_status_delta, character_status_updates, narrative_state, state_modifiers, character_overrides, anonymous_chars, orchestrator_note = summary_future.result()
            if wiki_future is not None:
                try:
                    wiki_context = wiki_future.result()
                except Exception as exc:
                    logger.warning(f"Wiki prefetch failed: {exc}")
                    wiki_context = ""

        # Store summarizer outputs raw. They reference the next scene's cast,
        # which may not exist in the current scene; normalization happens in
        # _load_scene once the next scene is loaded.
        self._next_scene_summaries = bridging_summaries
        self._next_scene_location_desc = finalized_descs.get(self.engine.loc.canonical_name, self.engine.loc.desc)
        self._next_scene_location_descs = finalized_descs
        self._next_scene_time = finalized_time
        self._next_scene_facts = facts
        # If the summarizer emitted a SYSTEM_STATE block, treat it as the complete
        # new state; otherwise carry the current state forward unchanged.
        self._next_scene_player_status = (
            player_status_delta if player_status_delta else dict(self.engine.player_status)
        )
        self._next_scene_free_status = dict(self.engine.free_status)
        self._next_scene_location_statuses = {
            loc.canonical_name: dict(loc.status)
            for loc in self._current_scene.location_pool
        }
        self._next_scene_state_modifiers = state_modifiers
        self._next_scene_character_status_updates = character_status_updates
        self._next_scene_character_overrides = character_overrides
        self._next_scene_anonymous_chars = anonymous_chars
        self._next_scene_orchestrator_note = orchestrator_note
        self._next_scene_wiki_context = wiki_context
        # Apply narrative state updates and mirror to wiki.
        if narrative_state:
            self._narrative_state.update(narrative_state)
            self._upsert_narrative_state()
        self._upsert_invented_facts(facts)
        total_chars = len(bridging_summaries)
        logger.info(f"Summarizer produced {total_chars} character summaries for transition to {next_scene_id}")

    def _normalize_state_modifiers(
        self,
        scene: Scene,
        modifiers: SceneStateModifiers,
    ) -> SceneStateModifiers:
        """Return a copy of *modifiers* with all name keys converted to canonical IDs."""
        normalized = SceneStateModifiers()
        normalized.player_status = dict(modifiers.player_status)
        normalized.world_status = dict(modifiers.world_status)
        normalized.narrative_state = dict(modifiers.narrative_state)

        for name, status in modifiers.character_status.items():
            char = scene.character_by_name(name)
            if char is not None:
                normalized.character_status[char.canonical_name] = status

        for name, status in modifiers.location_status.items():
            loc = scene.location_by_name(name)
            if loc is not None:
                normalized.location_status[loc.canonical_name] = status

        for name, sprite_entry in modifiers.sprites.items():
            char = scene.character_by_name(name)
            if char is None:
                continue
            entry = dict(sprite_entry)
            visible_to = set(entry.get("visible_to", []))
            resolved_visible_to = {
                observer.canonical_name
                for observer_name in visible_to
                if (observer := scene.character_by_name(observer_name)) is not None
            }
            entry["visible_to"] = list(resolved_visible_to)
            normalized.sprites[char.canonical_name] = entry

        return normalized

    def _apply_state_modifiers(
        self,
        scene: Scene,
        modifiers: SceneStateModifiers,
    ) -> None:
        """Apply mechanical state modifiers to the freshly-loaded scene."""
        if not modifiers:
            return

        if modifiers.player_status:
            self.engine._player_status = dict(modifiers.player_status)
        if modifiers.world_status:
            self.engine._free_status = dict(modifiers.world_status)

        for name, status in modifiers.character_status.items():
            char = scene.character_by_name(name)
            if char is not None:
                char.status = dict(status)

        for name, status in modifiers.location_status.items():
            loc = scene.location_by_name(name)
            if loc is not None:
                loc.status = dict(status)

        for name, sprite_entry in modifiers.sprites.items():
            char = scene.character_by_name(name)
            if char is None:
                continue
            sprite_name = sprite_entry.get("sprite", "")
            visible_to = set(sprite_entry.get("visible_to", []))
            if sprite_name == "hidden":
                char.hidden = True
                char.current_sprite = "none"
                char.visible_to = visible_to
            elif sprite_name == "none":
                char.hidden = False
                char.current_sprite = "none"
                char.visible_to = visible_to
            elif sprite_name:
                char.hidden = False
                char.current_sprite = sprite_name
                char.visible_to = visible_to

    def _upsert_narrative_state(self) -> None:
        """Mirror the story-level narrative state into the orchestrator wiki."""
        if self.db is None or not self._narrative_state:
            return
        try:
            import json
            self.db.upsert(
                "orchestrator_wiki",
                ids=["story:state"],
                documents=[json.dumps(self._narrative_state, ensure_ascii=False)],
                metadatas=[{"topic": "story:state", "importance": "critical", "trust": 1.0}],
            )
            logger.debug("Mirrored narrative state to orchestrator_wiki")
        except Exception as exc:
            logger.warning(f"Failed to mirror narrative state: {exc}")

    def _upsert_invented_facts(self, facts: list[dict[str, Any]]) -> None:
        """Persist invented facts from the summarizer into the orchestrator wiki."""
        if not facts or self.db is None:
            return
        for idx, fact in enumerate(facts):
            statement = fact.get("fact", "").strip()
            if not statement:
                continue
            trust = float(fact.get("trust", 0.0))
            source = fact.get("source", "").strip()
            topic = f"invented_fact_{idx:03d}"
            content = statement
            if source:
                content += f"\nSource: {source}"
            try:
                self.db.upsert(
                    "orchestrator_wiki",
                    ids=[topic],
                    documents=[content],
                    metadatas=[{"topic": topic, "importance": "notable", "trust": trust}],
                )
                logger.info(f"Upserted invented fact '{topic}' with trust {trust}")
            except Exception as exc:
                logger.warning(f"Failed to upsert invented fact: {exc}")

    def _generate_finalize_turn(self, next_scene_id: str) -> None:
        """Generate an opening narrator beat that justifies state changes.

        The finalize turn runs after the summarizer and before the next scene's
        orchestrator turn. It narrates major location (and later time) changes
        so the player sees a smooth transition.
        """
        if not self._current_scene or not self.engine.loc:
            return
        if not self._next_scene_location_desc and not self._next_scene_time:
            return
        changes = []
        if self._next_scene_location_desc and self._next_scene_location_desc.strip() != self.engine.loc.desc.strip():
            changes.append(f"location: {self.engine.loc.name}")
        if self._next_scene_time and self._next_scene_time != self.engine.world_time:
            changes.append(f"time: {self._next_scene_time}")
        if not changes:
            return

        system_prompt = f"""IMPORTANT: Write in {self._language} only!
You are the Narrator. Write a single short sentence (maximum 25 words) describing
how the scene has changed between scenes. Do NOT use meta-language, do NOT
address the player directly, and do NOT include dialogue."""

        previous_time = self.engine.world_time or "unspecified"
        updated_time = self._next_scene_time or previous_time
        user_prompt = f"""Previous location description:
{self.engine.loc.desc}

Updated location description:
{self._next_scene_location_desc}

Previous time: {previous_time}
Updated time: {updated_time}

Write one sentence describing the change."""

        try:
            narrator_name = (
                self.current_scene.narrator.name if self.current_scene else None
            )
            result = self.client.complete(
                role=GameRole.NARRATOR,
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                stream=False,
                name=narrator_name,
            )
            text = result.content.strip()
            if text:
                self._finalize_turn_text = text
                self._finalize_turn_changes = {
                    "location": self.engine.loc.name,
                    "next_scene": next_scene_id,
                }
                logger.info(f"Finalize turn generated: {text[:80]}")
        except Exception as exc:
            logger.debug(f"Finalize turn generation failed: {exc}")

    def run(
        self,
        get_user_input: Callable[[str, list[str]], str] | None = None,
        debug_console: DebugConsole | None = None,
    ) -> None:
        """Execute the full story from the initial scene onward.

        This is a convenience wrapper around :meth:`start` and :meth:`step`.
        For CLI-driven playback use the state-machine API directly.

        :param get_user_input: Callback that receives a prompt string and a
            list of suggestions, and returns the player's typed input.
            Defaults to a simple ``input()`` wrapper.
        :param debug_console: Optional debug console.
        """
        if get_user_input is None:
            get_user_input = self._get_user_input

        self.engine.set_debug_console(debug_console)
        self.start()

        while not self.finished:
            if (
                debug_console
                and debug_console.auto_pause
                and self._state == "running"
                and not self.engine.needs_player_input
            ):
                self._debug_pause(debug_console)

            result = self.step()

            if result.event == "scene_loaded" and result.scene is not None:
                print(f"\n=== Scene: {result.scene.id} ===")
                print(f"Location: {result.scene.starting_location.name}")
                print(f"Characters: {[c.name for c in result.scene.starting_characters]}\n")

            elif result.event == "turn" and result.output:
                print(result.output)
                print()

            elif result.event == "needs_player_input":
                suggestions = result.suggestions or []
                while True:
                    text = get_user_input(
                        f"{self.current_scene.player.name}> ", suggestions
                    )
                    stripped = text.strip()
                    if stripped.startswith(("/", ":")):
                        if debug_console is not None:
                            self._debug_pause(debug_console, noshell=stripped[1:])
                        else:
                            print("Debug console not available. Start with --debug-console.")
                        continue
                    break
                self.submit_player_input(text)

        print("\n=== Story Complete ===")
        print(f"Scenes visited: {self._scene_history}")

    def _debug_pause(self, debug_console: DebugConsole, noshell: str = "") -> None:
        """Enter the debug console with the current engine state."""
        if self.current_scene is None:
            return
        debug_console.pause(
            scene=self.current_scene,
            ctx=self.engine.ctx,
            here_chars=self.engine.here_chars,
            away_chars=self.engine.away_chars,
            loc=self.engine.loc or self.current_scene.starting_location,
            decision=self.engine.last_decision,
            noshell=noshell,
        )

    def _get_user_input(self, prompt: str, suggestions: list[str]) -> str:
        """Display *prompt* and read a line from stdin.

        :param prompt: Text to display before the cursor.
        :param suggestions: Orchestrator-provided suggestions (unused by the
            default implementation but available for custom front-ends).
        :return: The player's input, or a default continuation string on EOF.
        """
        print(prompt, end="")
        try:
            return input()
        except EOFError:
            return "[OOC: continue]"

    def jump_to(self, scene_id: str) -> StoryStep:
        """Abandon the current scene and jump directly to *scene_id*.

        :param scene_id: Scene identifier (filename without extension).
        :return: The loaded scene step.
        :raises RuntimeError: If the scene file does not exist.
        """
        path = self._resolve_scene_path(scene_id)
        if not path.exists():
            raise RuntimeError(f"Scene '{scene_id}' not found at {path}")
        self._current_path = path
        self._state = "loading"
        self._skipped_scene = True
        return self._load_scene()

    def _resolve_scene_path(self, scene_id: str) -> Path:
        """Map a scene identifier to a ``.toml`` file path.

        Scene files are looked up in the story directory.

        :param scene_id: Scene identifier (filename without extension).
        :return: Resolved file path.
        """
        return self._story_dir / f"{scene_id}.toml"
