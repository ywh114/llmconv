"""Story discovery for the webclient title screen."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ara.config import AraSettings
from ara.utils.logger import get_logger

logger = get_logger(__name__)

def discover_stories(plot_dir: Path | None = None) -> list[dict[str, Any]]:
    """Scan *plot_dir* for subdirectories containing ``ini_scene.toml``.

    :return: List of story metadata dicts with keys ``id``, ``title``,
        ``description``, ``author``, ``first_scene``.
    """
    base = plot_dir or AraSettings().plot_path
    stories: list[dict[str, Any]] = []
    if not base.exists():
        return stories
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        ini_path = subdir / "ini_scene.toml"
        if not ini_path.exists():
            continue
        try:
            with ini_path.open("rb") as f:
                data = tomllib.load(f)
            stories.append({
                "id": subdir.name,
                "title": data.get("title", subdir.name),
                "description": data.get("description", ""),
                "author": data.get("author", ""),
                "first_scene": data.get("first_scene", ""),
                "path": str(ini_path),
                "opening_text": data.get("opening_text", ""),
            })
        except Exception as exc:
            logger.warning(f"Failed to read {ini_path}: {exc}")
    return stories
