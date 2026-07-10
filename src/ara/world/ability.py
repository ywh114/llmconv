"""Lazy grammar-based ability generation for the fortune system.

Re-uses the generic grammar engine from ``title`` but with ability-specific
slot definitions, directories, and templates.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from ara.config import AraSettings
from ara.world.title import (
    LEVELS,
    _apply_expose,
    _load_toml,
    _resolve_level,
    build_grammar,
    expand,
    expand_all,
    title_case,
)

_ALWAYS_LOAD = {"generic", "numbers", "templates"}
_SLOTS = {"verb", "noun", "prefix", "suffix", "adj", "adj_sup"}


def _ability_dirs(
    story: str | None = None,
    config: AraSettings | None = None,
) -> tuple[Path, Path | None]:
    settings = config or AraSettings()
    global_dir = settings.fortune_path(None) / "ability"
    if story:
        story_dir = settings.fortune_path(story) / "ability"
        if story_dir.exists():
            return story_dir, global_dir
    return global_dir, None


def list_ability_flavors(
    story: str | None = None,
    config: AraSettings | None = None,
) -> list[str]:
    """Return the sorted list of available ability flavor names."""
    primary, fallback = _ability_dirs(story, config)
    stems: set[str] = set()
    for directory in (primary, fallback):
        if directory is None:
            continue
        if directory.exists():
            for path in directory.glob("*.toml"):
                if path.stem not in _ALWAYS_LOAD:
                    stems.add(path.stem)
    return sorted(stems)


def categorized_ability_flavors(
    story: str | None = None,
    config: AraSettings | None = None,
) -> dict[str, list[str]]:
    """Return ability flavors grouped by category.

    Categories come from each TOML's ``category`` key.
    """
    primary, fallback = _ability_dirs(story, config)
    by_cat: dict[str, list[str]] = {}
    for directory in (primary, fallback):
        if directory is None:
            continue
        if not directory.exists():
            continue
        for path in directory.glob("*.toml"):
            stem = path.stem
            if stem in _ALWAYS_LOAD:
                continue
            cat = "other"
            try:
                data = _load_toml(stem, primary, fallback)
                cat = data.get("category", "other")
            except Exception:
                pass
            by_cat.setdefault(cat, []).append(stem)
    for k in by_cat:
        by_cat[k].sort()
    return by_cat


def _load_templates(
    primary: Path,
    fallback: Path | None,
    level: str | None = None,
    exact: bool = False,
) -> list[str]:
    templates_data = _load_toml("templates", primary, fallback)
    if level:
        if exact:
            return list(templates_data.get(level, []))
        templates: list[str] = []
        for lvl in LEVELS:
            templates.extend(templates_data.get(lvl, []))
            if lvl == level:
                break
        return templates
    templates = []
    for lvl in LEVELS:
        templates.extend(templates_data.get(lvl, []))
    return templates


def load_ability_grammar(
    story: str | None = None,
    config: AraSettings | None = None,
    flavors: list[str] | str | None = None,
    slot_sources: dict[str, list[str]] | None = None,
    cull_sources: bool = True,
) -> dict:
    primary, fallback = _ability_dirs(story, config)
    available = list_ability_flavors(story, config)
    if flavors is None:
        selected = available
    elif isinstance(flavors, str):
        selected = [flavors]
    else:
        selected = list(flavors)
    unknown = [s for s in selected if s not in available]
    if unknown:
        raise ValueError(f"Unknown ability flavor(s): {', '.join(unknown)}")
    return build_grammar(
        selected, slot_sources or {}, primary, fallback, cull_sources=cull_sources
    )


def generate_ability(
    story: str | None = None,
    config: AraSettings | None = None,
    template: str | None = None,
    flavors: list[str] | str | None = None,
    level: str | int | None = "2",
    slot_sources: dict[str, list[str]] | None = None,
    required_slots: list[str] | set[str] | None = None,
    cull_sources: bool = True,
) -> str:
    primary, fallback = _ability_dirs(story, config)
    level_name, exact = _resolve_level(level)
    templates = _load_templates(primary, fallback, level_name, exact)
    if required_slots:
        required = set(required_slots)
        templates = [
            tmpl
            for tmpl in templates
            if required.issubset(set(re.findall(r"\{([^}+]+)\}", tmpl)))
        ]
        if not templates:
            raise ValueError(
                f"No templates contain all required slots: {', '.join(sorted(required))}"
            )
    grammar = load_ability_grammar(
        story=story,
        config=config,
        flavors=flavors,
        slot_sources=slot_sources,
        cull_sources=cull_sources,
    )
    if template is not None:
        tmpl = template
    else:
        tmpl = random.choice(templates)
    return expand(tmpl, grammar)
