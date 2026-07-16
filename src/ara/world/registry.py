"""Ephemeral per-session registry for characters and locations.

The registry stores live-state documents for every character and location
referenced by the current story's plot TOMLs.  Documents live in ChromaDB so
they are searchable by vector similarity, but they are rebuilt each session and
are intentionally not snapshotted by the save/load system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tomllib

from ara.config import AraSettings
from ara.memory.chroma import ChromaStore
from ara.world.i18n import normalize_language
from ara.utils.logger import get_logger

logger = get_logger(__name__)

from ara.world.ids import stable_uuid as _stable_uuid


CHARACTER_COLLECTION = "ara_characters"
LOCATION_COLLECTION = "ara_locations"


def _card_names(card: dict[str, Any]) -> dict[str, str]:
    """Return the merged [names] table plus the card name as a fallback 'en'."""
    names: dict[str, str] = dict(card.get("names", {}))
    display = card.get("name", "")
    if display and "en" not in names:
        names["en"] = display
    return names


def _serialize_names(names: dict[str, str]) -> str:
    """Serialize a names table for ChromaDB metadata storage."""
    return json.dumps(names, sort_keys=True, ensure_ascii=False)


def _meta_names(meta: dict[str, Any]) -> dict[str, str]:
    """Return the names table from registry metadata, deserializing if needed."""
    raw = meta.get("names", {})
    if isinstance(raw, str):
        try:
            return dict(json.loads(raw))
        except Exception:
            return {}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _serialize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested metadata values so ChromaDB can store them."""
    out = dict(meta)
    for key in ("names", "live_state"):
        value = out.get(key)
        if isinstance(value, dict):
            out[key] = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return out


def _deserialize_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Restore nested metadata values after reading them from ChromaDB."""
    out = dict(meta)
    for key in ("names", "live_state"):
        value = out.get(key)
        if isinstance(value, str):
            try:
                out[key] = dict(json.loads(value))
            except Exception:
                out[key] = {}
    return out


def _match_name(name: str, language: str, card: dict[str, Any], canonical: str) -> str | None:
    """Return the canonical name if *name* matches this card in *language*.

    The directory name is the canonical ID; the card's ``name`` field and
    ``[names]`` table are display-name aliases.
    """
    if not canonical:
        return None
    display = card.get("name", "")
    if name == canonical:
        return canonical
    if display and name == display:
        return canonical
    names = _card_names(card)
    if name == names.get(language, ""):
        return canonical
    # Also accept English matches regardless of requested language.
    if language != "en" and name == names.get("en", ""):
        return canonical
    return None


class AssetRegistry:
    """Indexes referenced characters/locations into ChromaDB each session.

    The registry is intentionally ephemeral.  On ``Story.start()`` and on
    load-from-save it crawls the plot directory, finds every referenced
    character/location, and upserts a document per canonical asset.  Live
    runtime state is merged in lazily via :meth:`sync_live_state`.
    """

    def __init__(self, config: AraSettings, db: ChromaStore) -> None:
        self.config = config
        self.db = db
        self._story_cc_dir: Path | None = None
        self._story_lc_dir: Path | None = None

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def index_story(self, story_dir: Path, language: str) -> None:
        """Rebuild the character and location registries from *story_dir*.

        Assets are resolved under ``data/assets/{cc,lc}/<story>/``.
        """
        language = normalize_language(language)
        story_name = story_dir.name
        self._story_cc_dir = self.config.characters_path(story_name)
        self._story_lc_dir = self.config.locations_path(story_name)
        char_names, loc_names = self._collect_references(story_dir)
        self._index_characters(char_names, language)
        self._index_locations(loc_names, language)

    def _collect_references(self, story_dir: Path) -> tuple[set[str], set[str]]:
        """Return sets of referenced character and location display names."""
        char_names: set[str] = set()
        loc_names: set[str] = set()
        for toml_path in story_dir.rglob("*.toml"):
            try:
                with toml_path.open("rb") as f:
                    data = tomllib.load(f)
            except Exception:
                continue
            char_data = data.get("character", {})
            if isinstance(char_data, dict):
                char_names.update(char_data.get("pool", []))
                char_names.update(char_data.get("sprites", {}).keys())
                char_names.update(char_data.get("titles", {}).keys())
                char_names.add(char_data.get("player", ""))
                char_names.add(char_data.get("narrator", ""))

            loc_data = data.get("location", {})
            if isinstance(loc_data, dict):
                loc_names.update(loc_data.get("pool", []))
                loc_names.add(loc_data.get("init", ""))
                loc_names.update(loc_data.get("descs", {}).keys())
                loc_names.update(loc_data.get("loading", {}).keys())

        char_names.discard("")
        loc_names.discard("")
        return char_names, loc_names

    def _index_characters(self, names: set[str], language: str) -> None:
        existing = self._load_existing(CHARACTER_COLLECTION)
        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for canonical, card, asset_dir in self._walk_characters(names, language):
            cid = _stable_uuid("character", canonical)
            meta = self._build_character_meta(canonical, card, asset_dir, language)
            # Preserve any live-state that already exists in this session.
            if cid in existing:
                meta["live_state"] = existing[cid].get("live_state", {})
            else:
                meta["live_state"] = {}
            docs.append(self._character_document(meta))
            ids.append(cid)
            metadatas.append(_serialize_meta(meta))

        if docs:
            self.db.upsert(CHARACTER_COLLECTION, ids=ids, documents=docs, metadatas=metadatas)
        logger.debug(f"Indexed {len(docs)} characters")

    def _index_locations(self, names: set[str], language: str) -> None:
        existing = self._load_existing(LOCATION_COLLECTION)
        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for canonical, card, asset_dir in self._walk_locations(names, language):
            lid = _stable_uuid("location", canonical)
            meta = self._build_location_meta(canonical, card, asset_dir, language)
            if lid in existing:
                meta["live_state"] = existing[lid].get("live_state", {})
            else:
                meta["live_state"] = {}
            docs.append(self._location_document(meta))
            ids.append(lid)
            metadatas.append(_serialize_meta(meta))

        if docs:
            self.db.upsert(LOCATION_COLLECTION, ids=ids, documents=docs, metadatas=metadatas)
        logger.debug(f"Indexed {len(docs)} locations")

    def _walk_characters(
        self, names: set[str], language: str
    ) -> list[tuple[str, dict[str, Any], Path]]:
        """Find asset directories matching the referenced *names*.

        Searches only the per-story ``data/assets/cc/<story>/`` directory.
        """
        results: list[tuple[str, dict[str, Any], Path]] = []
        unresolved = set(names)
        story_dir = self._story_cc_dir
        if story_dir is None or not story_dir.exists():
            return results

        # First pass: match by canonical name or [names] alias.
        for char_dir in story_dir.iterdir():
            if not char_dir.is_dir():
                continue
            card_path = char_dir / "card.toml"
            if not card_path.exists():
                continue
            try:
                with card_path.open("rb") as f:
                    card = tomllib.load(f)
            except Exception:
                continue
            # Use the directory name as the canonical name if the card omits one.
            if not card.get("name"):
                card["name"] = char_dir.name
            for name in list(unresolved):
                matched = _match_name(name, language, card, char_dir.name)
                if matched:
                    results.append((matched, card, char_dir))
                    unresolved.discard(name)
                    break

        if unresolved:
            logger.warning(f"Unresolved characters: {sorted(unresolved)}")
        return results

    def _walk_locations(
        self, names: set[str], language: str
    ) -> list[tuple[str, dict[str, Any], Path]]:
        """Find asset directories matching the referenced *names*.

        Searches only the per-story ``data/assets/lc/<story>/`` directory.
        """
        results: list[tuple[str, dict[str, Any], Path]] = []
        unresolved = set(names)
        story_dir = self._story_lc_dir
        if story_dir is None or not story_dir.exists():
            return results

        for loc_dir in story_dir.iterdir():
            if not loc_dir.is_dir():
                continue
            card_path = loc_dir / "card.toml"
            if not card_path.exists():
                continue
            try:
                with card_path.open("rb") as f:
                    card = tomllib.load(f)
            except Exception:
                continue
            # Use the directory name as the canonical name if the card omits one.
            if not card.get("name"):
                card["name"] = loc_dir.name
            for name in list(unresolved):
                matched = _match_name(name, language, card, loc_dir.name)
                if matched:
                    results.append((matched, card, loc_dir))
                    unresolved.discard(name)
                    break

        if unresolved:
            logger.warning(f"Unresolved locations: {sorted(unresolved)}")
        return results

    # ------------------------------------------------------------------ #
    # Metadata builders
    # ------------------------------------------------------------------ #
    def _build_character_meta(
        self, canonical: str, card: dict[str, Any], asset_dir: Path, language: str
    ) -> dict[str, Any]:
        names = _card_names(card)
        return {
            "id": _stable_uuid("character", canonical),
            "canonical_name": canonical,
            "display_name": names.get(language, canonical),
            "names": _serialize_names(names),
            "language": language,
            "asset_dir": str(asset_dir),
            "summary": card.get("summary", ""),
            "personality": card.get("personality", ""),
            "scenario": card.get("scenario", ""),
            "greeting_message": card.get("greeting_message", ""),
            "example_messages": card.get("example_messages", ""),
        }

    def _build_location_meta(
        self, canonical: str, card: dict[str, Any], asset_dir: Path, language: str
    ) -> dict[str, Any]:
        names = _card_names(card)
        return {
            "id": _stable_uuid("location", canonical),
            "canonical_name": canonical,
            "display_name": names.get(language, canonical),
            "names": _serialize_names(names),
            "language": language,
            "asset_dir": str(asset_dir),
            "description": card.get("description", ""),
            "lore": card.get("lore", ""),
            "loading_background": card.get("loading_background", ""),
        }

    def _character_document(self, meta: dict[str, Any]) -> str:
        """Build the searchable text for a character document."""
        parts = [
            f"Name: {meta['canonical_name']}",
            f"Display: {meta['display_name']}",
        ]
        if meta.get("summary"):
            parts.append(f"Summary: {meta['summary']}")
        if meta.get("personality"):
            parts.append(f"Personality: {meta['personality']}")
        if meta.get("example_messages"):
            parts.append(f"Examples:\n{meta['example_messages']}")
        live = meta.get("live_state", {})
        if live.get("scratch"):
            parts.append(f"Scratch: {live['scratch']}")
        if live.get("title"):
            parts.append(f"Title: {live['title']}")
        return "\n\n".join(parts)

    def _location_document(self, meta: dict[str, Any]) -> str:
        """Build the searchable text for a location document."""
        parts = [
            f"Name: {meta['canonical_name']}",
            f"Display: {meta['display_name']}",
        ]
        if meta.get("description"):
            parts.append(f"Description: {meta['description']}")
        if meta.get("lore"):
            parts.append(f"Lore: {meta['lore']}")
        live = meta.get("live_state", {})
        if live.get("desc"):
            parts.append(f"Current: {live['desc']}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Existing-document merging
    # ------------------------------------------------------------------ #
    def _load_existing(self, collection_name: str) -> dict[str, dict[str, Any]]:
        """Return existing documents keyed by id, parsed from metadata."""
        try:
            raw = self.db.get_all(collection_name)
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        result: dict[str, dict[str, Any]] = {}
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        if not isinstance(ids, list) or not isinstance(metadatas, list):
            return {}
        for cid, meta in zip(ids, metadatas):
            if meta:
                result[cid] = _deserialize_meta(dict(meta))
        return result

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def get_character(self, name: str, language: str) -> dict[str, Any] | None:
        """Return registry metadata for a character by display/canonical name."""
        language = normalize_language(language)
        existing = self._load_existing(CHARACTER_COLLECTION)
        for meta in existing.values():
            if meta.get("canonical_name") == name:
                return meta
            names = _meta_names(meta)
            if name == names.get(language, "") or name == names.get("en", ""):
                return meta
        return None

    def get_location(self, name: str, language: str) -> dict[str, Any] | None:
        """Return registry metadata for a location by display/canonical name."""
        language = normalize_language(language)
        existing = self._load_existing(LOCATION_COLLECTION)
        for meta in existing.values():
            if meta.get("canonical_name") == name:
                return meta
            names = _meta_names(meta)
            if name == names.get(language, "") or name == names.get("en", ""):
                return meta
        return None

    # ------------------------------------------------------------------ #
    # Live-state sync
    # ------------------------------------------------------------------ #
    def sync_live_state(
        self,
        characters: list[Any] | None = None,
        locations: list[Any] | None = None,
    ) -> None:
        """Upsert the current runtime state of live objects into ChromaDB.

        This is intended to be called lazily before a search or at end-of-scene.
        Static card fields are left untouched; only the ``live_state`` metadata
        is updated.
        """
        if characters:
            self._sync_collection(
                CHARACTER_COLLECTION,
                characters,
                lambda c: {
                    "scratch": getattr(c.scratch, "text", ""),
                    "title": getattr(c, "title", ""),
                    "status": json.dumps(getattr(c, "status", {})),
                    "current_sprite": getattr(c, "current_sprite", ""),
                },
            )
        if locations:
            self._sync_collection(
                LOCATION_COLLECTION,
                locations,
                lambda loc: {
                    "desc": getattr(loc, "desc", ""),
                    "current_background": getattr(loc, "current_background", ""),
                },
            )

    def _sync_collection(
        self,
        collection_name: str,
        objects: list[Any],
        extractor: Any,
    ) -> None:
        existing = self._load_existing(collection_name)
        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for obj in objects:
            canonical = getattr(obj, "canonical_name", "") or getattr(obj, "name", "")
            if not canonical:
                continue
            cid = _stable_uuid(collection_name.removeprefix("ara_"), canonical)
            if cid not in existing:
                continue
            meta = existing[cid].copy()
            meta["live_state"] = extractor(obj)
            docs.append(
                self._character_document(meta)
                if collection_name == CHARACTER_COLLECTION
                else self._location_document(meta)
            )
            ids.append(cid)
            metadatas.append(_serialize_meta(meta))
        if docs:
            self.db.upsert(collection_name, ids=ids, documents=docs, metadatas=metadatas)

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search_characters(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Vector-search characters."""
        return self._search(CHARACTER_COLLECTION, query, n_results)

    def search_locations(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Vector-search locations."""
        return self._search(LOCATION_COLLECTION, query, n_results)

    def _search(
        self, collection_name: str, query: str, n_results: int
    ) -> list[dict[str, Any]]:
        result = self.db.query(collection_name, [query], n_results=n_results)
        metadatas = result.get("metadatas") or [[]]
        return [dict(m) for m in metadatas[0] if m]
