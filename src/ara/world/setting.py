"""World-setting TOML loader.

Settings are collections of facts organised into arbitrary categories that are
automatically loaded into the orchestrator wiki when a story starts.  No category
is hard-coded; any top-level table array (or single table) in the TOML becomes a
fact category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from ara.config import AraSettings
from ara.utils.logger import get_logger

logger = get_logger(__name__)


# Metadata keys that are consumed by the loader itself and never treated as
# fact categories.
_RESERVED_KEYS = {"id", "name", "summary"}


def _item_id(item: dict[str, Any], index: int) -> str:
    """Derive a stable identifier for a category item.

    Uses common identifying fields if present, falling back to the item index.
    """
    for key in ("name", "topic", "period", "id", "title"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return str(index)


@dataclass
class WorldSetting:
    """Parsed world setting TOML.

    :ivar id: Short machine identifier (e.g. ``azur_lane``).
    :ivar name: Human-readable name.
    :ivar summary: Short overview of the setting.
    :ivar categories: Mapping from category name to a list of fact items.
        Each item is a dict whose fields are stored as the wiki document text.
    """

    id: str
    name: str = ""
    summary: str = ""
    categories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def wiki_entries(self) -> dict[str, str]:
        """Return wiki topic -> document mapping for this setting.

        Topics are namespaced by the setting id so multiple settings can coexist.
        """
        entries: dict[str, str] = {}
        prefix = f"world:{self.id}"

        if self.summary:
            entries[f"{prefix}:summary"] = self.summary

        for category_name, items in self.categories.items():
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                item_id = _item_id(item, idx)
                entries[f"{prefix}:{category_name}:{item_id}"] = _format_item(item)

        return entries


def _format_item(item: dict[str, Any]) -> str:
    """Format a TOML item dict as a short human-readable paragraph."""
    parts = [f"{k}: {v}" for k, v in item.items() if v is not None]
    return "\n".join(parts)


def load_world_setting(path: Path) -> WorldSetting:
    """Load a world setting from a TOML file.

    Every top-level array of tables (``[[category]]``) or inline table becomes a
    fact category.  Reserved keys ``id``, ``name``, and ``summary`` are used for
    the setting metadata.

    :param path: Path to the ``.toml`` file.
    :return: Parsed :class:`WorldSetting`.
    :raises FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"World setting not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)

    world_id = data.get("id", path.stem)

    categories: dict[str, list[dict[str, Any]]] = {}
    for key, value in data.items():
        if key in _RESERVED_KEYS:
            continue
        if isinstance(value, list):
            categories[key] = [item for item in value if isinstance(item, dict)]
        elif isinstance(value, dict):
            categories[key] = [value]
        else:
            # Non-table values are not fact categories; skip them.
            logger.debug(f"Skipping non-table world setting key '{key}'")

    return WorldSetting(
        id=world_id,
        name=data.get("name", world_id),
        summary=data.get("summary", ""),
        categories=categories,
    )


def resolve_world_setting_path(setting_id: str, config: AraSettings, story: str | None = None) -> Path:
    """Return the path for a world setting, preferring per-story overrides.

    Resolution order:
    1. ``data/assets/world/<story>/<setting_id>.toml`` if *story* is given.
    2. ``data/assets/world/<setting_id>.toml`` (global fallback).

    :param setting_id: World setting identifier (file stem).
    :param config: Application settings.
    :param story: Optional story id for per-story lookup.
    :return: Resolved path (not guaranteed to exist).
    """
    if story:
        story_path = config.world_path(story) / f"{setting_id}.toml"
        if story_path.exists():
            return story_path
    return config.world_path() / f"{setting_id}.toml"
