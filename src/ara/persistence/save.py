"""VN-style save/load system for Ara stories.

Saves capture the full runtime state of a :class:`~ara.world.story.Story` so
that play can be resumed later from the exact same point.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ara.config import AraSettings
from ara.memory.chroma import ChromaStore
from ara.utils.logger import get_logger
from ara.memory.knowledge import CharacterMemory
from ara.world.character import Character, Importance, create_anonymous_character, load_character
from ara.llm.context import ConversationContext

from ara.world.i18n import normalize_language
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene, load_location
from ara.world.story import Story

logger = get_logger(__name__)

SAVE_VERSION = 5
MAX_SLOTS = 99


def _slices_to_json(slices: list[slice]) -> list[list[int | None]]:
    return [[sl.start, sl.stop] for sl in slices]


def _slices_from_json(data: list[list[int | None]]) -> list[slice]:
    return [slice(item[0], item[1]) for item in data]


def _ctx_to_dict(ctx: ConversationContext) -> dict[str, Any]:
    return {
        "injected_context": list(ctx.injected_context),
        "context": list(ctx.context),
        "head": dict(ctx.head) if ctx.head else None,
        "seen_entities": {
            k: _slices_to_json(v) for k, v in ctx.seen_entities.items()
        },
        "present_entities": list(ctx.present_entities),
    }


def _ctx_from_dict(data: dict[str, Any]) -> ConversationContext:
    entities = list(data["seen_entities"].keys())
    ctx = ConversationContext(*entities)
    ctx.injected_context = list(data.get("injected_context", []))
    ctx.context = list(data.get("context", []))
    ctx.head = data.get("head")
    ctx.seen_entities = {
        k: _slices_from_json(v) for k, v in data["seen_entities"].items()
    }
    ctx.present_entities = set(data.get("present_entities", []))
    return ctx


def _decision_to_dict(dec: TurnDecision | None) -> dict[str, Any] | None:
    if dec is None:
        return None
    return {
        "next_char": dec.next_char.canonical_name,
        "directive": dec.directive,
        "suggestions": list(dec.suggestions),
        "entering_chars": [c.canonical_name for c in dec.entering_chars],
        "exiting_chars": [c.canonical_name for c in dec.exiting_chars],
        "switch_location": dec.switch_location.canonical_name if dec.switch_location else None,
        "edit_location": dec.edit_location,
        "next_scene": dec.next_scene,
        "change_sprite": dict(dec.change_sprite),
        "switch_background": dec.switch_background,
        "spawn_anonymous": list(dec.spawn_anonymous),
        "set_time": dec.set_time,
        "system_changes": dict(dec.system_changes),
        "response_mode": dec.response_mode,
    }


def _decision_from_dict(
    data: dict[str, Any] | None, scene: Scene
) -> TurnDecision | None:
    if data is None:
        return None
    return TurnDecision(
        next_char=scene.character_by_canonical(data["next_char"]),
        directive=data["directive"],
        suggestions=list(data.get("suggestions", [])),
        entering_chars={scene.character_by_canonical(n) for n in data.get("entering_chars", [])},
        exiting_chars={scene.character_by_canonical(n) for n in data.get("exiting_chars", [])},
        switch_location=scene.location_by_canonical(data.get("switch_location")) if data.get("switch_location") else None,
        edit_location=data.get("edit_location", ""),
        next_scene=data.get("next_scene"),
        change_sprite=dict(data.get("change_sprite", {})),
        switch_background=data.get("switch_background", ""),
        spawn_anonymous=list(data.get("spawn_anonymous", [])),
        set_time=data.get("set_time", ""),
        system_changes=dict(data.get("system_changes", {})),
        response_mode=data.get("response_mode", "outer"),
    )


def _character_snapshot(char: Character, db: ChromaStore | None) -> dict[str, Any]:
    """Return a serialisable snapshot of a character's runtime state."""
    mem_docs: list[str] = []
    if isinstance(char.memory, CharacterMemory) and db is not None:
        try:
            raw = db.get_all(char.memory.collection_name)
            mem_docs = raw.get("documents", []) or []
        except Exception as exc:
            logger.warning(f"Could not read memory for {char.canonical_name}: {exc}")
    return {
        "id": str(char.id),
        "canonical_name": char.canonical_name,
        "name": char.name,
        "scratch": char.scratch.text,
        "scratch_prev": char.scratch.prev_text,
        "prev_scene_summary": char.prev_scene_summary,
        "current_sprite": char.current_sprite,
        "status": dict(char.status),
        "names": dict(char.names),
        "title": char.title,
        "hidden": char.hidden,
        "visible_to": list(char.visible_to),
        "inner_log": list(char.inner_log),
        "memory_documents": mem_docs,
        "card_overrides": dict(char.card_overrides),
    }


def _apply_character_snapshot(char: Character, cd: dict[str, Any], db: ChromaStore | None) -> None:
    """Restore runtime state from a character snapshot."""
    char.scratch.text = cd.get("scratch", char.scratch.text)
    char.scratch.prev_text = cd.get("scratch_prev", char.scratch.prev_text)
    char.prev_scene_summary = cd.get("prev_scene_summary", "")
    char.current_sprite = cd.get("current_sprite", char.current_sprite)
    char.status = dict(cd.get("status", {}))
    char.names = dict(cd.get("names", {}))
    char.title = cd.get("title", "")
    char.hidden = cd.get("hidden", False)
    char.visible_to = set(cd.get("visible_to", []))
    char.inner_log = list(cd.get("inner_log", []))
    char.card_overrides = dict(cd.get("card_overrides", {}))
    mem_docs = cd.get("memory_documents", [])
    if mem_docs and isinstance(char.memory, CharacterMemory) and db is not None:
        try:
            coll = db.collection(char.memory.collection_name)
            existing = coll.get()
            existing_ids = existing.get("ids", []) or []
            if existing_ids:
                coll.delete(ids=existing_ids)
            char.memory.add_conversation(mem_docs)
        except Exception as exc:
            logger.warning(f"Could not restore memory for {char.name}: {exc}")


def _location_snapshot(loc: Location) -> dict[str, Any]:
    """Return a serialisable snapshot of a location's runtime state."""
    return {
        "canonical_name": loc.canonical_name,
        "name": loc.name,
        "desc": loc.desc,
        "status": dict(loc.status),
        "names": dict(loc.names),
        "current_background": loc.current_background,
    }


def _apply_location_snapshot(loc: Location, ld: dict[str, Any]) -> None:
    """Restore runtime state from a location snapshot."""
    loc.desc = ld.get("desc", loc.desc)
    loc.status = dict(ld.get("status", {}))
    loc.names = dict(ld.get("names", {}))
    loc.current_background = ld.get("current_background", loc.current_background)


def _restore_live_character(
    canonical: str, cd: dict[str, Any], story: Story, db: ChromaStore | None
) -> Character | None:
    """Rebuild a live-cache character from assets and apply a snapshot."""
    if db is None:
        return None
    config = story.config
    meta = story.registry.get_character(canonical, story._language)
    if meta is not None:
        asset_dir = Path(meta["asset_dir"])
    else:
        asset_dir = config.characters_path(story._story_dir.name) / canonical
    if not asset_dir.exists():
        logger.warning(f"Cannot restore live character {canonical}: asset dir not found")
        return None
    try:
        char = load_character(asset_dir, db, story._language)
    except Exception as exc:
        logger.warning(f"Failed to restore live character {canonical}: {exc}")
        return None
    _apply_character_snapshot(char, cd, db)
    return char


def _restore_live_location(
    canonical: str, ld: dict[str, Any], story: Story
) -> Location | None:
    """Rebuild a live-cache location from assets and apply a snapshot."""
    config = story.config
    meta = story.registry.get_location(canonical, story._language)
    if meta is not None:
        asset_dir = Path(meta["asset_dir"])
    else:
        asset_dir = config.locations_path(story._story_dir.name) / canonical
    if not asset_dir.exists():
        logger.warning(f"Cannot restore live location {canonical}: asset dir not found")
        return None
    try:
        loc = load_location(asset_dir, language=story._language)
    except Exception as exc:
        logger.warning(f"Failed to restore live location {canonical}: {exc}")
        return None
    _apply_location_snapshot(loc, ld)
    return loc


@dataclass
class SaveInfo:
    """Metadata about a save slot."""

    slot: int
    story_id: str
    scene_id: str | None
    timestamp: str
    scene_history: list[str] = field(default_factory=list)


class SaveManager:
    """Manages save files for Ara stories.

    Saves are stored as JSON files under ``<settings.saves_path>/<story_id>/slot_<N>.json``.
    Each save captures a full snapshot of story + engine + character state.
    """

    def __init__(self, settings: AraSettings) -> None:
        """Create a save manager.

        :param settings: Application settings (used for ``saves_path``).
        """
        self.settings = settings
        self.base_dir = settings.saves_path

    # ------------------------------------------------------------------ #
    # Discovery
    # ------------------------------------------------------------------ #

    def list_saves(self, story_id: str) -> list[SaveInfo]:
        """Return metadata for all existing save slots for *story_id*."""
        saves: list[SaveInfo] = []
        story_dir = self.base_dir / story_id
        if not story_dir.exists():
            return saves
        for path in sorted(story_dir.glob("slot_*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                saves.append(
                    SaveInfo(
                        slot=int(path.stem.split("_")[1]),
                        story_id=data.get("story_id", story_id),
                        scene_id=data.get("current_scene_id"),
                        timestamp=data.get("timestamp", ""),
                        scene_history=data.get("scene_history", []),
                    )
                )
            except Exception as exc:
                logger.warning(f"Skipping corrupt save {path}: {exc}")
        return saves

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #

    def save(self, story: Story, slot: int, queue: list[dict[str, Any]] | None = None) -> Path:
        """Persist a full snapshot of *story* into *slot*.

        :param story: Active story instance.
        :param slot: Save slot number (1–99).
        :param queue: Optional event queue to persist alongside story state.
        :return: Path to the written save file.
        """
        snapshot = self._build_snapshot(story, queue=queue)
        return self.save_snapshot(snapshot, slot)

    def save_snapshot(self, snapshot: dict[str, Any], slot: int) -> Path:
        """Write a pre-built snapshot dict to *slot*.

        :param snapshot: Snapshot dictionary (e.g. from ``_build_snapshot``).
        :param slot: Save slot number (1–99).
        :return: Path to the written save file.
        """
        if not 1 <= slot <= MAX_SLOTS:
            raise ValueError(f"Slot must be 1–{MAX_SLOTS}, got {slot}")

        story_id = snapshot.get("story_id", "")
        story_dir = self.base_dir / story_id
        story_dir.mkdir(parents=True, exist_ok=True)
        path = story_dir / f"slot_{slot:02d}.json"

        with path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved slot {slot} for {story_id} → {path}")
        return path

    def _build_snapshot(
        self,
        story: Story,
        queue: list[dict[str, Any]] | None = None,
        include_archive: bool = True,
    ) -> dict[str, Any]:
        """Build a serialisable snapshot of *story*.

        :param include_archive: When ``False``, omit the append-only
            ``archived_scene_snapshots`` telescope.  Callers that rebuild
            snapshots frequently can skip it and carry the live list by
            reference; it is only needed when a save is written to disk.
        """
        engine = story.engine
        scene = story.current_scene
        db = story.db

        # Character snapshots
        char_snapshots: list[dict[str, Any]] = []
        if scene is not None:
            for char in scene.character_pool:
                char_snapshots.append(_character_snapshot(char, db))

        # Live-cache snapshots preserve off-screen characters/locations too.
        live_character_snapshots = [
            _character_snapshot(c, db) for c in story._live_characters.values()
        ]
        live_location_snapshots = [
            _location_snapshot(loc) for loc in story._live_locations.values()
        ]

        # Runtime-spawned anonymous characters (not present in scene TOML)
        anonymous_snapshots: list[dict[str, Any]] = []
        if scene is not None:
            for char in scene.character_pool:
                if char.importance == Importance.ANONYMOUS:
                    anonymous_snapshots.append(
                        {
                            "name": char.name,
                            "description": char.card_fields.get("summary", ""),
                            "sprite": char.card_fields.get("sprite", ""),
                            "status": dict(char.status),
                            "current_sprite": char.current_sprite,
                            "hidden": char.hidden,
                            "visible_to": list(char.visible_to),
                        }
                    )

        # Story history documents
        story_history_docs: list[dict[str, Any]] = []
        try:
            raw = db.get_all("story_history")
            ids = raw.get("ids", []) or []
            docs = raw.get("documents", []) or []
            metas = raw.get("metadatas", []) or []
            for i, doc in enumerate(docs):
                story_history_docs.append(
                    {
                        "id": ids[i] if i < len(ids) else str(i),
                        "document": doc,
                        "metadata": metas[i] if i < len(metas) else {},
                    }
                )
        except Exception as exc:
            logger.warning(f"Could not read story_history: {exc}")

        # Orchestrator wiki documents
        wiki_docs: list[dict[str, Any]] = []
        try:
            raw = db.get_all("orchestrator_wiki")
            ids = raw.get("ids", []) or []
            docs = raw.get("documents", []) or []
            metas = raw.get("metadatas", []) or []
            for i, doc in enumerate(docs):
                wiki_docs.append(
                    {
                        "id": ids[i] if i < len(ids) else str(i),
                        "document": doc,
                        "metadata": metas[i] if i < len(metas) else {},
                    }
                )
        except Exception as exc:
            logger.warning(f"Could not read orchestrator_wiki: {exc}")

        # Engine state
        engine_state: dict[str, Any] = {}
        if engine is not None:
            engine_state = {
                "_running": engine._running,
                "_needs_player_input": engine._needs_player_input,
                "_turn_count": engine._turn_count,
                "_speaker_history": list(engine._speaker_history),
                "_directives_log": {
                    c.canonical_name: d for c, d in (engine._directives_log or {}).items()
                },
                "_next_scene": engine._next_scene,
                "_prev_char": engine._prev_char.canonical_name if engine._prev_char else None,
                "_here_chars": [c.canonical_name for c in engine._here_chars],
                "_away_chars": [c.canonical_name for c in engine._away_chars],
                "_loc": engine._loc.canonical_name if engine._loc else None,
                "_ctx": _ctx_to_dict(engine._ctx) if engine._ctx else None,
                "_last_decision": _decision_to_dict(engine._last_decision),
                "_player_status": dict(engine._player_status),
                "_world_time": engine._world_time,
                "_mechanical_changelog": list(engine._mechanical_changelog),
                "_story_state": dict(engine._story_state),
                "_pending_attempts": list(engine._pending_attempts),
                "_free_status": dict(engine._free_status),
                "_location_statuses": (
                    {
                        loc.canonical_name: dict(loc.status)
                        for loc in scene.location_pool
                    }
                    if scene
                    else {}
                ),
                "_prefetched_wiki": engine.orchestrator.prefetched_wiki,
                "_orchestrator_scratch_text": engine.orchestrator.scratch.text,
                "_orchestrator_scratch_prev_text": engine.orchestrator.scratch.prev_text,
                "_canonical_index": getattr(engine, "_canonical_index", 0),
                "_canonical_pending_choices": (
                    list(engine._canonical_pending_choices)
                    if getattr(engine, "_canonical_pending_choices", None) is not None
                    else None
                ),
            }

        snapshot: dict[str, Any] = {
            "version": SAVE_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "story_id": story._story_dir.name,
            "initial_scene_path": str(story.initial_scene_path),
            "scene_history": list(story.scene_history),
            "current_scene_id": scene.id if scene else None,
            "current_path": str(story._current_path) if story._current_path else None,
            "finalize_turn_text": story._finalize_turn_text,
            "finalize_turn_changes": dict(story._finalize_turn_changes),
            "story_state": {
                "_state": story._state,
                "_skipped_scene": story._skipped_scene,
                "_next_scene_summaries": story._next_scene_summaries,
                "_next_scene_location_desc": story._next_scene_location_desc,
                "_next_scene_location_descs": story._next_scene_location_descs,
                "_next_scene_time": story._next_scene_time,
                "_next_scene_player_status": story._next_scene_player_status,
                "_next_scene_wiki_context": story._next_scene_wiki_context,
                "_character_status": dict(story._character_status),
                "_narrative_state": dict(story._narrative_state),
                "_world_id": story._world_id,
                "_loaded_settings": list(story._loaded_settings),
                "_language": story._language,
            },
            "location_descs": {
                loc.canonical_name: loc.desc for loc in (scene.location_pool if scene else [])
            },
            "engine_state": engine_state,
            "characters": char_snapshots,
            "anonymous_characters": anonymous_snapshots,
            "live_characters": live_character_snapshots,
            "live_locations": live_location_snapshots,
            "story_history": story_history_docs,
            "orchestrator_wiki": wiki_docs,
            "queue": queue if queue is not None else [],
        }
        if include_archive:
            snapshot["archived_scene_snapshots"] = list(story._archived_scene_snapshots)
        return snapshot

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load(self, story: Story, slot: int) -> list[dict[str, Any]]:
        """Restore *story* from *slot*.

        The story must already be initialised with the same
        ``initial_scene_path`` that was used when the save was created.

        :param story: Story instance to restore into.
        :param slot: Save slot number.
        :return: The restored event queue.
        :raises FileNotFoundError: If the slot does not exist.
        """
        story_id = story._story_dir.name
        path = self.base_dir / story_id / f"slot_{slot:02d}.json"
        if not path.exists():
            raise FileNotFoundError(f"Save slot {slot} not found for {story_id}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("version") != SAVE_VERSION:
            raise ValueError(
                f"Unsupported save version {data.get('version')}; expected {SAVE_VERSION}"
            )

        self._apply_snapshot(story, data)
        logger.info(f"Loaded slot {slot} for {story_id}")
        return list(data.get("queue", []))

    def _apply_snapshot(self, story: Story, data: dict[str, Any]) -> None:
        db = story.db

        # Chroma is transient: wipe any collections left over from the current
        # runtime so the snapshot is restored into a clean store.
        if db is not None:
            db.clear_all_collections()

        # Restore story_history to ChromaDB
        for item in data.get("story_history", []):
            try:
                db.upsert(
                    "story_history",
                    ids=[item["id"]],
                    documents=[item["document"]],
                    metadatas=[item.get("metadata", {})],
                )
            except Exception as exc:
                logger.warning(f"Could not restore story_history doc {item.get('id')}: {exc}")

        # Restore orchestrator wiki to ChromaDB
        for item in data.get("orchestrator_wiki", []):
            try:
                db.upsert(
                    "orchestrator_wiki",
                    ids=[item["id"]],
                    documents=[item["document"]],
                    metadatas=[item.get("metadata", {})],
                )
            except Exception as exc:
                logger.warning(f"Could not restore wiki doc {item.get('id')}: {exc}")

        # Reset story to a clean idle state before applying snapshot
        story._prev_scene = None
        story._current_scene = None
        story._state = "idle"
        story.engine._running = False
        story._archived_scene_snapshots = list(data.get("archived_scene_snapshots", []))

        # Set story state before loading scene
        saved_history = list(data.get("scene_history", []))
        story_state = data.get("story_state", {})
        saved_state = story_state.get("_state", "running")
        saved_current_path = data.get("current_path")
        saved_finalize_text = data.get("finalize_turn_text", "")
        saved_finalize_changes = dict(data.get("finalize_turn_changes", {}))
        story._skipped_scene = story_state.get("_skipped_scene", False)
        story._next_scene_summaries = dict(story_state.get("_next_scene_summaries", {}))
        story._next_scene_location_desc = story_state.get("_next_scene_location_desc", "")
        story._next_scene_location_descs = dict(story_state.get("_next_scene_location_descs", {}))
        story._next_scene_time = story_state.get("_next_scene_time", "")
        story._next_scene_player_status = dict(story_state.get("_next_scene_player_status", {}))
        story._next_scene_wiki_context = story_state.get("_next_scene_wiki_context", "")
        story._character_status = dict(story_state.get("_character_status", {}))
        story._narrative_state = dict(story_state.get("_narrative_state", {}))
        story._world_id = story_state.get("_world_id", "")
        story._loaded_settings = set(story_state.get("_loaded_settings", []))
        story._language = normalize_language(
            story_state.get("_language", story._language)
        )

        # Restore live caches so Scene.load reuses existing objects for any
        # characters/locations that appear now or in the future.
        for cd in data.get("live_characters", []):
            canonical = cd.get("canonical_name", cd.get("name", ""))
            if not canonical or canonical in story._live_characters:
                continue
            char = _restore_live_character(canonical, cd, story, db)
            if char is not None:
                story._live_characters[canonical] = char
        for ld in data.get("live_locations", []):
            canonical = ld.get("canonical_name", ld.get("name", ""))
            if not canonical or canonical in story._live_locations:
                continue
            loc = _restore_live_location(canonical, ld, story)
            if loc is not None:
                story._live_locations[canonical] = loc

        # Load current scene
        current_scene_id = data.get("current_scene_id")
        if current_scene_id:
            story._current_path = story._resolve_scene_path(current_scene_id)
            story._state = "loading"
            # Force-load the scene (this appends to _scene_history)
            step = story._load_scene()
            # Restore the exact saved history to prevent duplicate append
            story._scene_history = saved_history
            if step.event == "story_complete":
                return
        else:
            # No scene was active: start fresh from the story's first scene
            story.start()
            story._scene_history = saved_history
            return

        scene = story.current_scene
        if scene is None:
            return

        # Restore location descriptions (must happen before engine restarts).
        saved_location_descs = data.get("location_descs", {})
        for loc in scene.location_pool:
            if loc.canonical_name in saved_location_descs:
                loc.desc = saved_location_descs[loc.canonical_name]

        # Restore runtime-spawned anonymous characters so they are available
        # for the engine here/away restore below.
        engine_state = data.get("engine_state", {})
        anon_here_names = set(engine_state.get("_here_chars", []))
        anon_away_names = set(engine_state.get("_away_chars", []))
        for anon_data in data.get("anonymous_characters", []):
            name = anon_data["name"]
            if scene.character_by_name(name) is not None:
                continue
            anon = create_anonymous_character(
                name=name,
                description=anon_data.get("description", ""),
                sprite=anon_data.get("sprite", ""),
            )
            anon.status = dict(anon_data.get("status", {}))
            anon.current_sprite = anon_data.get("current_sprite", "")
            anon.hidden = anon_data.get("hidden", False)
            anon.visible_to = set(anon_data.get("visible_to", []))
            scene.character_pool.add(anon)
            if name in anon_away_names:
                # Away extras are in the pool but not in the starting set.
                pass
            else:
                scene.starting_characters.add(anon)

        # Restore character state
        char_data = {c.get("canonical_name", c["name"]): c for c in data.get("characters", [])}
        for char in scene.character_pool:
            cd = char_data.get(char.canonical_name)
            if cd is None:
                continue
            _apply_character_snapshot(char, cd, db)

        # Restore engine state on top of the freshly-started engine
        engine_state = data.get("engine_state", {})
        if engine_state:
            engine = story.engine
            engine._running = engine_state.get("_running", True)
            engine._needs_player_input = engine_state.get("_needs_player_input", False)
            engine._turn_count = engine_state.get("_turn_count", 0)
            engine._speaker_history = list(engine_state.get("_speaker_history", []))
            saved_directives = engine_state.get("_directives_log") or {}
            engine._directives_log = {}
            for char in scene.character_pool:
                if char.canonical_name in saved_directives:
                    engine._directives_log[char] = saved_directives[char.canonical_name]
            engine._next_scene = engine_state.get("_next_scene")

            prev_char_name = engine_state.get("_prev_char")
            if prev_char_name:
                engine._prev_char = scene.character_by_canonical(prev_char_name)

            here_names = set(engine_state.get("_here_chars", []))
            engine._here_chars = {c for c in scene.character_pool if c.canonical_name in here_names}
            away_names = set(engine_state.get("_away_chars", []))
            engine._away_chars = {c for c in scene.character_pool if c.canonical_name in away_names}

            loc_name = engine_state.get("_loc")
            if loc_name:
                engine._loc = scene.location_by_canonical(loc_name)

            ctx_data = engine_state.get("_ctx")
            if ctx_data:
                engine._ctx = _ctx_from_dict(ctx_data)

            last_decision_data = engine_state.get("_last_decision")
            if last_decision_data:
                engine._last_decision = _decision_from_dict(last_decision_data, scene)

            engine._player_status = dict(engine_state.get("_player_status", {}))
            engine._world_time = engine_state.get("_world_time", "")
            engine._mechanical_changelog = list(engine_state.get("_mechanical_changelog", []))
            engine._story_state = dict(engine_state.get("_story_state", {}))
            engine._pending_attempts = list(
                engine_state.get("_pending_attempts")
                or engine_state.get("_pending_orchestrator_notes", [])
            )
            engine._free_status = dict(engine_state.get("_free_status", {}))
            saved_location_statuses = engine_state.get("_location_statuses", {})
            for loc in scene.location_pool:
                if loc.canonical_name in saved_location_statuses:
                    loc.status = dict(saved_location_statuses[loc.canonical_name])
            engine.orchestrator.prefetched_wiki = engine_state.get("_prefetched_wiki", "")
            engine.orchestrator.scratch.text = engine_state.get(
                "_orchestrator_scratch_text", engine.orchestrator.scratch.text
            )
            engine.orchestrator.scratch.prev_text = engine_state.get(
                "_orchestrator_scratch_prev_text", engine.orchestrator.scratch.prev_text
            )

            # Restore canonical-script progress so scripted scenes do not replay
            # from the first event after load.
            engine._canonical_index = engine_state.get("_canonical_index", 0)
            pending_choices = engine_state.get("_canonical_pending_choices")
            engine._canonical_pending_choices = (
                list(pending_choices) if pending_choices is not None else None
            )

        # Restore the story state-machine position and any pending finalize turn.
        if saved_state == "finalizing":
            story._state = "finalizing"
            if saved_current_path:
                story._current_path = Path(saved_current_path)
            elif story.engine._next_scene:
                story._current_path = story._resolve_scene_path(story.engine._next_scene)
        else:
            story._state = saved_state if saved_state not in ("idle", "loading") else "running"
        story._finalize_turn_text = saved_finalize_text
        story._finalize_turn_changes = saved_finalize_changes

    # ------------------------------------------------------------------ #
    # Delete
    # ------------------------------------------------------------------ #

    def delete(self, story_id: str, slot: int) -> None:
        """Delete save *slot* for *story_id*."""
        path = self.base_dir / story_id / f"slot_{slot:02d}.json"
        if path.exists():
            path.unlink()
            logger.info(f"Deleted save slot {slot} for {story_id}")

    def delete_all(self, story_id: str) -> None:
        """Delete all saves for *story_id*."""
        story_dir = self.base_dir / story_id
        if story_dir.exists():
            shutil.rmtree(story_dir)
            logger.info(f"Deleted all saves for {story_id}")
