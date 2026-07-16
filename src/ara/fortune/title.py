"""Lazy grammar-based title generation for the fortune system."""

from __future__ import annotations

import random
import re
import tomllib
from pathlib import Path

from ara.config import AraSettings


from ara.fortune.tokens import (
    expand as _expand,
    expand_all as _expand_all,
    expand_all_traced as _expand_all_traced,
    expand_traced as _expand_traced,
    title_case as _title_case,
)


LEVELS = ['simple', 'moderate', 'complex', 'insane']
ALWAYS_LOAD = {'generic', 'numbers', 'templates'}
GENERIC_SLOTS = {
    'rank',
    'class',
    'noun',
    'entity',
    'adj',
    'adj_sup',
    'prefix',
    'place',
    'number',
    'ordinal',
    'roman_numeral',
    'suffix',
}


def title_dirs(
    story: str | None = None,
    config: AraSettings | None = None,
) -> tuple[Path, Path | None]:
    """Return (primary_dir, fallback_dir) for title assets.

    If a story-specific ``title/`` directory exists, it is the primary and the
    global directory becomes the fallback. Otherwise only the global directory
    is used.
    """
    settings = config or AraSettings()
    global_dir = settings.fortune_path(None) / 'title'
    if story:
        story_dir = settings.fortune_path(story) / 'title'
        if story_dir.exists():
            return story_dir, global_dir
    return global_dir, None


def load_toml(name: str, primary: Path, fallback: Path | None = None) -> dict:
    for directory in (primary, fallback):
        if directory is None:
            continue
        path = directory / f'{name}.toml'
        if path.exists():
            with path.open('rb') as f:
                return tomllib.load(f)
    raise FileNotFoundError(f'Title TOML not found: {name}.toml')


def list_title_flavors(
    story: str | None = None,
    config: AraSettings | None = None,
) -> list[str]:
    """Return the sorted list of available flavor names."""
    primary, fallback = title_dirs(story, config)
    stems: set[str] = set()
    for directory in (primary, fallback):
        if directory is None:
            continue
        if directory.exists():
            for path in directory.glob('*.toml'):
                if path.stem not in ALWAYS_LOAD:
                    stems.add(path.stem)
    return sorted(stems)


def categorized_title_flavors(
    story: str | None = None,
    config: AraSettings | None = None,
) -> dict[str, list[str]]:
    """Return title flavors grouped by category (from each TOML's ``category`` key)."""
    primary, fallback = title_dirs(story, config)
    by_cat: dict[str, list[str]] = {}
    for directory in (primary, fallback):
        if directory is None:
            continue
        if not directory.exists():
            continue
        for path in directory.glob('*.toml'):
            stem = path.stem
            if stem in ALWAYS_LOAD:
                continue
            cat = 'other'
            try:
                data = load_toml(stem, primary, fallback)
                cat = data.get('category', 'other')
            except Exception:
                pass
            by_cat.setdefault(cat, []).append(stem)
    for k in by_cat:
        by_cat[k].sort()
    return by_cat


def apply_expose(grammar: dict) -> dict:
    """Populate generic slots from internal slots using [[expose]] mappings."""
    for entry in grammar.get('expose', []):
        generic_slot = entry['slot']
        internal_slots = entry['from']
        combined: list[dict] = []
        for internal in internal_slots:
            items = grammar.pop(internal, [])
            if internal != generic_slot:
                combined.extend(items)
            grammar[internal] = items
        if generic_slot in grammar:
            grammar[generic_slot] = grammar[generic_slot] + combined
        else:
            grammar[generic_slot] = combined
    return grammar


def expand(
    pattern: str,
    grammar: dict,
    depth: int = 0,
    max_depth: int = 10,
) -> str:
    """Expand a title template into a concrete, title-cased string."""
    return _expand(pattern, grammar, depth, max_depth)


def expand_traced(
    pattern: str,
    grammar: dict,
    depth: int = 0,
    max_depth: int = 10,
    _parent: str = '',
) -> tuple[str, list[dict]]:
    """Expand a title template into a concrete string with provenance trace."""
    return _expand_traced(pattern, grammar, depth, max_depth, _parent)


def expand_all(
    pattern: str,
    grammar: dict,
    depth: int = 0,
    max_depth: int = 10,
) -> list[str]:
    """Enumerate every expansion of a template."""
    return _expand_all(pattern, grammar, depth, max_depth)


def expand_all_traced(
    pattern: str,
    grammar: dict,
    depth: int = 0,
    max_depth: int = 10,
) -> list[tuple[str, list[dict]]]:
    """Enumerate every expansion of a template with provenance traces."""
    return _expand_all_traced(pattern, grammar, depth, max_depth)


def title_case(text: str) -> str:
    """Title-case a generated title, preserving known acronyms and small words."""
    return _title_case(text)


def resolve_level(
    level: str | int | None,
) -> tuple[str | None, bool]:
    """Return (level_name, exact)."""
    if level is None:
        return None, False
    if isinstance(level, int):
        if 0 <= level < len(LEVELS):
            return LEVELS[level], False
        raise ValueError(
            f'Level index {level} out of range (0-{len(LEVELS) - 1}).'
        )
    raw = str(level).strip()
    exact = False
    if raw.endswith('!'):
        exact = True
        raw = raw[:-1]
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(LEVELS):
            return LEVELS[idx], exact
        raise ValueError(
            f'Level index {idx} out of range (0-{len(LEVELS) - 1}).'
        )
    if raw == 'all':
        return None, False
    if raw in LEVELS:
        return raw, exact
    raise ValueError(f'Unknown level: {raw}. Use {", ".join(LEVELS)} or 0-3.')


def load_templates(
    primary: Path,
    fallback: Path | None,
    level: str | None = None,
    exact: bool = False,
) -> list[str]:
    templates_data = load_toml('templates', primary, fallback)
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


def _deduplicate_entries(entries: list[dict]) -> list[dict]:
    """Remove duplicate value entries from a slot, keeping the best one.

    Priority:
    1. More metadata fields (excluding value/patterns/_source).
    2. Base form wins: if A.value == B.noun_form|gerund|past|..., B wins.
    3. Otherwise keep the first encountered entry.
    """
    form_fields = (
        'noun_form',
        'gerund',
        'past',
        'plural',
        'superlative',
        'comparative',
        'possessive',
    )

    def _meta_count(entry: dict) -> int:
        return sum(
            1 for k in entry if k not in ('value', 'patterns', '_source')
        )

    def _is_base_form(base: dict, derived: dict) -> bool:
        base_value = str(base.get('value', '')).lower()
        for field in form_fields:
            if base_value == str(derived.get(field, '')).lower():
                return True
        return False

    best: dict[str, dict] = {}
    first_index: dict[str, int] = {}
    result: list[dict] = []

    for i, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or 'patterns' in entry
            or 'value' not in entry
        ):
            result.append(entry)
            continue
        key = str(entry['value']).lower()
        if key not in best:
            best[key] = entry
            first_index[key] = i
            continue
        current = best[key]
        if _meta_count(entry) > _meta_count(current):
            best[key] = entry
        elif _meta_count(entry) == _meta_count(current):
            if _is_base_form(current, entry):
                best[key] = entry
            elif not _is_base_form(entry, current):
                # Tie with no form relationship; keep current (first encountered).
                pass
    # Emit the chosen entries in first-encounter order.
    for i, entry in enumerate(entries):
        if (
            not isinstance(entry, dict)
            or 'patterns' in entry
            or 'value' not in entry
        ):
            continue
        key = str(entry['value']).lower()
        if first_index.get(key) == i:
            result.append(best[key])
    return result


def _cap_sources(
    entries: list[dict],
    slot: str,
    max_share: float = 0.4,
) -> list[dict]:
    """Trim any single source so it cannot dominate a generic slot.

    The cap defaults to ``max_share`` of the total slot size (minimum 1).
    Sampling is seeded randomly so repeated runs can produce different
    grammars.  The process repeats until no source exceeds the cap relative
    to the final slot size.
    """
    if not entries:
        return entries
    if not entries:
        return entries
    total = len(entries)
    cap = max(1, int(total * max_share))
    by_source: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        src = entry.get('_source', '')
        by_source.setdefault(src, []).append(i)
    # Capping is only meaningful when there are enough sources for 25%
    # to be a meaningful share. With one or two sources the user has
    # explicitly narrowed the pool, so leave it untouched.
    if len(by_source) <= 2:
        return entries
    if all(len(indices) <= cap for indices in by_source.values()):
        return entries
    keep_indices: set[int] = set()
    rng = random.Random()
    for indices in by_source.values():
        if len(indices) > cap:
            keep_indices.update(rng.sample(indices, cap))
        else:
            keep_indices.update(indices)
    return [entries[i] for i in sorted(keep_indices)]


def build_grammar(
    selected_flavors: list[str],
    slot_sources: dict[str, list[str]] | dict[str, list[list[str]]],
    primary: Path,
    fallback: Path | None,
    cull_sources: bool = True,
) -> dict:
    """Build a merged grammar from selected flavors with optional slot restrictions.

    *slot_sources* maps slot names to either a flat list of source names
    (union) or a list of priority groups. Each group is a list of sources
    tried in first-match order; groups are unioned.
    """
    # Normalize: flat list -> single group
    _slot_groups: dict[str, list[list[str]]] = {}
    for slot, sources in slot_sources.items():
        if sources and isinstance(sources[0], list):
            _slot_groups[slot] = sources  # type: ignore[arg-type]
        else:
            _slot_groups[slot] = [sources]  # type: ignore[arg-type]

    source_names = ['generic', 'numbers'] + selected_flavors
    grammars: dict[str, dict] = {}
    for name in source_names:
        grammars[name] = apply_expose(load_toml(name, primary, fallback))

    flat: dict[str, list] = {}
    for g in grammars.values():
        for key, value in g.items():
            if key in ('expose', 'category'):
                continue
            if isinstance(value, list):
                flat.setdefault(key, []).extend(value)

    generic_slots: set[str] = set()
    for g in grammars.values():
        for entry in g.get('expose', []):
            generic_slots.add(entry['slot'])

    all_slots: set[str] = set()
    for g in grammars.values():
        all_slots.update(g.keys())
    all_slots.discard('expose')
    all_slots.discard('category')

    merged: dict[str, list] = {'__generic_slots__': frozenset(generic_slots)}
    for slot in all_slots:
        entries: list[dict] = []
        if slot in generic_slots:
            groups = _slot_groups.get(slot)
            if groups is None:
                flavor_sources = [
                    s for s in source_names if s not in ('generic', 'numbers')
                ]
                has_flavor = any(
                    slot in grammars[s] and grammars[s][slot]
                    for s in flavor_sources
                )
                groups = [flavor_sources if has_flavor else source_names]
            else:
                # Filter out generic/numbers from each group
                groups = [
                    [s for s in grp if s not in ('generic', 'numbers')]
                    for grp in groups
                ]

            for grp in groups:
                for source in grp:
                    if source.endswith('!'):
                        entries.append(
                            {
                                'value': source[:-1],
                                'prefixable': False,
                                'suffixible': False,
                                '_source': 'literal',
                            }
                        )
                    elif source.startswith('@'):
                        # @flavor or @flavor:group
                        rest = source[1:]
                        if ':' in rest:
                            fname, group = rest.split(':', 1)
                            if fname in grammars and group in grammars[fname]:
                                tagged = [
                                    dict(e, _source=source)
                                    for e in grammars[fname][group]
                                ]
                                entries.extend(tagged)
                            else:
                                raise ValueError(
                                    f"Unknown @reference '{source}' for slot '{slot}'."
                                )
                        else:
                            if rest in grammars and slot in grammars[rest]:
                                tagged = [
                                    dict(e, _source=source)
                                    for e in grammars[rest][slot]
                                ]
                                entries.extend(tagged)
                            else:
                                raise ValueError(
                                    f"Unknown @flavor '{rest}' for slot '{slot}'."
                                )
                    elif source.endswith(':'):
                        ikey = source[:-1]
                        if ikey in flat:
                            tagged = [
                                dict(e, _source=source.rstrip(':'))
                                for e in flat[ikey]
                            ]
                            entries.extend(tagged)
                        else:
                            raise ValueError(
                                f"Unknown internal group '{ikey}' for slot '{slot}'."
                            )
                    elif (
                        _slot_groups.get(slot) is not None
                        and source in flat
                        and flat[source]
                        and isinstance(flat[source][0], dict)
                        and 'patterns' in flat[source][0]
                    ):
                        # When the user explicitly restricts a slot to a bare
                        # technique group name (melee, area, status, ...), resolve
                        # it across all flavors, not just the flavor file with the
                        # same name.  Flavor-specific selection is still available
                        # via @melee, @area, etc.
                        tagged = [dict(e, _source=source) for e in flat[source]]
                        entries.extend(tagged)
                    elif source in grammars:
                        if slot in grammars[source]:
                            tagged = [
                                dict(e, _source=source)
                                for e in grammars[source][slot]
                            ]
                            entries.extend(tagged)
                            # If a merged internal group with the same name exists,
                            # include it too (it spans all flavours). Delivery
                            # groups (melee/ranged/area/status) only contribute to
                            # the "technique" slot.
                            if source in flat and slot == 'technique':
                                tagged2 = [
                                    dict(e, _source=':' + source)
                                    for e in flat[source]
                                ]
                                entries.extend(tagged2)
                    elif source in flat:
                        tagged = [dict(e, _source=source) for e in flat[source]]
                        entries.extend(tagged)
                    else:
                        valid = [
                            s
                            for s in source_names
                            if s in grammars
                            and slot in grammars[s]
                            and grammars[s][slot]
                        ]
                        hint = ''
                        if valid:
                            hint = f' Available: {", ".join(valid)}'
                        raise ValueError(
                            f"Unknown source '{source}' for slot '{slot}'.{hint}"
                        )
            # Fallback: if restricted sources yielded nothing, use all sources
            if not entries and _slot_groups.get(slot):
                for source in source_names:
                    if source in grammars and slot in grammars[source]:
                        tagged = [
                            dict(e, _source=source)
                            for e in grammars[source][slot]
                        ]
                        entries.extend(tagged)
        else:
            for source in source_names:
                if source in grammars and slot in grammars[source]:
                    tagged = [
                        dict(e, _source=source) for e in grammars[source][slot]
                    ]
                    entries.extend(tagged)
        if entries:
            deduped = _deduplicate_entries(entries)
            if slot in generic_slots and cull_sources:
                deduped = _cap_sources(deduped, slot)
            merged[slot] = deduped

    return merged


def cull_grammar(grammar: dict) -> dict:
    """Return a copy of *grammar* with generic slots source-capped.

    This lets callers build one full grammar and cull it fresh for each
    individual title/ability generation instead of once per batch.
    """
    generic_slots = grammar.get('__generic_slots__', frozenset())
    if not generic_slots:
        return grammar
    culled: dict[str, list] = {}
    for slot, entries in grammar.items():
        if slot == '__generic_slots__':
            culled[slot] = entries
        elif slot in generic_slots and isinstance(entries, list):
            culled[slot] = _cap_sources(list(entries), slot)
        else:
            culled[slot] = list(entries)
    return culled


def load_title_grammar(
    story: str | None = None,
    config: AraSettings | None = None,
    flavors: list[str] | str | None = None,
    slot_sources: dict[str, list[str]] | None = None,
    cull_sources: bool = True,
) -> dict:
    """Load and merge title flavor grammars.

    :param flavors: Flavor name(s) to load. ``None`` loads every available flavor.
    :param slot_sources: Optional per-slot source restrictions.
    :param cull_sources: If False, skip source-cap culling.
    """
    primary, fallback = title_dirs(story, config)
    available = list_title_flavors(story, config)

    if flavors is None:
        selected = available
    elif isinstance(flavors, str):
        selected = [flavors]
    else:
        selected = list(flavors)

    unknown = [s for s in selected if s not in available]
    if unknown:
        raise ValueError(f'Unknown title flavor(s): {", ".join(unknown)}')

    return build_grammar(
        selected,
        slot_sources or {},
        primary,
        fallback,
        cull_sources=cull_sources,
    )


def generate_title(
    story: str | None = None,
    config: AraSettings | None = None,
    template: str | None = None,
    flavors: list[str] | str | None = None,
    level: str | int | None = '2',
    slot_sources: dict[str, list[str]] | None = None,
    required_slots: list[str] | set[str] | None = None,
    cull_sources: bool = True,
) -> str:
    """Generate a random title.

    :param template: Specific template string or ``None`` for random.
    :param flavors: Flavor name(s) to use, or ``None`` for all.
    :param level: Complexity level name/index, optionally suffixed with ``!`` for exact.
    :param slot_sources: Per-slot source restrictions.
    :param required_slots: Only use templates containing all of these slots.
    """
    primary, fallback = title_dirs(story, config)
    level_name, exact = resolve_level(level)
    templates = load_templates(primary, fallback, level_name, exact)

    if required_slots:
        required = set(required_slots)
        templates = [
            tmpl
            for tmpl in templates
            if required.issubset(set(re.findall(r'\{([^}+]+)\}', tmpl)))
        ]
        if not templates:
            raise ValueError(
                f'No templates contain all required slots: {", ".join(sorted(required))}'
            )

    grammar = load_title_grammar(
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
