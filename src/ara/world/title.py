"""Lazy grammar-based title generation for the fortune system."""

from __future__ import annotations

import random
import re
import tomllib
from pathlib import Path
from typing import Any

from ara.config import AraSettings


LEVELS = ["simple", "moderate", "complex", "insane"]
_ALWAYS_LOAD = {"generic", "numbers", "templates"}
_GENERIC_SLOTS = {
    "rank",
    "class",
    "noun",
    "adj",
    "adj_sup",
    "prefix",
    "place",
    "thing",
    "number",
    "ordinal",
    "roman_numeral",
    "suf",
}
# Slot ordering: place/domain must expand before sufs so article stripping sees {suf}
_PLACE_FIRST = {"place", "domain"}
_SUF_LAST = {"suf"}

# Slots whose entries may carry an "article" field (the/a/an)
_ARTICLE_SLOTS = {"place", "domain"}

# Words that expect their object to keep its article (e.g. "of the North", not "of North").
# Includes conjunctions so that list items keep their own articles
# (e.g. "of the Kuiper Belt and the World Map").
_KEEP_ARTICLE_AFTER = frozenset({
    "of", "from", "in", "on", "at", "to", "by", "for", "with", "without",
    "upon", "over", "under", "against", "beyond", "through", "between",
    "among", "within", "across", "behind", "beside", "near",
    "inside", "outside", "toward", "towards", "about", "into", "onto",
    "before", "after", "along", "around", "below", "beneath", "during",
    "throughout", "via", "past", "despite", "except", "like",
    "and", "or", "nor", "yet",
})


def _should_keep_article(before_text: str) -> bool:
    """Return True if a slot's article should be kept, based on the preceding text.

    A slot's article is normally dropped when it follows a regular word or
    another slot placeholder — but preserved when the preceding word is a
    conjunction (``and``, ``or``, etc.) or preposition (``of``, ``from``, …).
    """
    if not before_text:
        return True
    tokens = before_text.split()
    last_token = tokens[-1] if tokens else before_text.rstrip()
    if not last_token:
        return True
    last_char = last_token[-1]
    if last_char.islower() or last_char == '}':
        return last_token.lower() in _KEEP_ARTICLE_AFTER
    return True


def _title_dirs(
    story: str | None = None,
    config: AraSettings | None = None,
) -> tuple[Path, Path | None]:
    """Return (primary_dir, fallback_dir) for title assets.

    If a story-specific ``title/`` directory exists, it is the primary and the
    global directory becomes the fallback. Otherwise only the global directory
    is used.
    """
    settings = config or AraSettings()
    global_dir = settings.fortune_path(None) / "title"
    if story:
        story_dir = settings.fortune_path(story) / "title"
        if story_dir.exists():
            return story_dir, global_dir
    return global_dir, None


def _load_toml(name: str, primary: Path, fallback: Path | None = None) -> dict:
    for directory in (primary, fallback):
        if directory is None:
            continue
        path = directory / f"{name}.toml"
        if path.exists():
            with path.open("rb") as f:
                return tomllib.load(f)
    raise FileNotFoundError(f"Title TOML not found: {name}.toml")


def list_title_flavors(
    story: str | None = None,
    config: AraSettings | None = None,
) -> list[str]:
    """Return the sorted list of available flavor names."""
    primary, fallback = _title_dirs(story, config)
    stems: set[str] = set()
    for directory in (primary, fallback):
        if directory is None:
            continue
        if directory.exists():
            for path in directory.glob("*.toml"):
                if path.stem not in _ALWAYS_LOAD:
                    stems.add(path.stem)
    return sorted(stems)


def categorized_title_flavors(
    story: str | None = None,
    config: AraSettings | None = None,
) -> dict[str, list[str]]:
    """Return title flavors grouped by category (from each TOML's ``category`` key)."""
    primary, fallback = _title_dirs(story, config)
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


def _apply_expose(grammar: dict) -> dict:
    """Populate generic slots from internal slots using [[expose]] mappings."""
    for entry in grammar.get("expose", []):
        generic_slot = entry["slot"]
        internal_slots = entry["from"]
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
    """Expand a title template into a concrete string."""
    if depth >= max_depth:
        return pattern

    result = pattern

    # Sort keys so place is expanded before sufs (article stripping depends on it)
    ordered_slots = sorted(
        grammar.keys(),
        key=lambda k: (2 if k in _SUF_LAST else 0 if k in _PLACE_FIRST else 1),
    )

    while True:
        replaced = False
        for slot in ordered_slots:
            placeholder = "{" + slot + "}"
            while placeholder in result:
                entries = grammar[slot]
                if not entries:
                    break
                entry = random.choice(entries)
                if isinstance(entry, dict) and "patterns" in entry:
                    sub_pattern = random.choice(entry["patterns"])
                    replacement = expand(sub_pattern, grammar, depth + 1, max_depth)
                elif isinstance(entry, dict) and "value" in entry:
                    value = entry["value"]
                    article = entry.get("article", "")
                    noun_form = entry.get("noun_form", "")
                    if not article and slot in _ARTICLE_SLOTS and value:
                        m = re.match(r'^(the|an?)\s+', value, re.IGNORECASE)
                        if m:
                            article = m.group(1)
                            value = value[m.end():]
                    if article and slot in _ARTICLE_SLOTS:
                        placeholder_pos = result.find(placeholder)
                        if not _should_keep_article(result[:placeholder_pos]):
                            skip_article = True
                        else:
                            skip_article = False
                            after_pos = placeholder_pos + len(placeholder)
                            if after_pos < len(result) and result[after_pos:].startswith("{suf"):
                                skip_article = True
                        if not skip_article:
                            value = article + " " + value
                    elif noun_form:
                        placeholder_pos = result.find(placeholder)
                        after_pos = placeholder_pos + len(placeholder)
                        if after_pos < len(result) and result[after_pos:].startswith(" of"):
                            value = noun_form
                    replacement = value
                else:
                    replacement = str(entry)
                result = result.replace(placeholder, replacement, 1)
                replaced = True
        if not replaced:
            break

    stripped = re.sub(r'^([Tt]he)\s(\S+)$', r'\2', result, 1)
    if stripped != result:
        result = stripped

    return result


def expand_traced(
    pattern: str,
    grammar: dict,
    depth: int = 0,
    max_depth: int = 10,
    _parent: str = "",
) -> tuple[str, list[dict]]:
    """Like :func:`expand` but returns ``(text, trace)`` where *trace* is a
    list of ``{"slot": str, "value": str, "source": str, "parent": str}`` dicts.
    *parent* is set when the slot was resolved via a pattern in a parent slot
    (e.g. ``place -> nato_name``).
    """
    if depth >= max_depth:
        return pattern, []

    trace: list[dict] = []

    result = pattern

    ordered_slots = sorted(
        grammar.keys(),
        key=lambda k: (2 if k in _SUF_LAST else 0 if k in _PLACE_FIRST else 1),
    )

    while True:
        replaced = False
        for slot in ordered_slots:
            placeholder = "{" + slot + "}"
            while placeholder in result:
                entries = grammar[slot]
                if not entries:
                    break
                entry = random.choice(entries)
                src = entry.get("_source", "?")
                if isinstance(entry, dict) and "patterns" in entry:
                    sub_pattern = random.choice(entry["patterns"])
                    replacement, sub_trace = expand_traced(
                        sub_pattern, grammar, depth + 1, max_depth,
                        _parent=slot,
                    )
                    trace.extend(sub_trace)
                elif isinstance(entry, dict) and "value" in entry:
                    value = entry["value"]
                    article = entry.get("article", "")
                    noun_form = entry.get("noun_form", "")
                    if not article and slot in _ARTICLE_SLOTS and value:
                        m = re.match(r'^(the|an?)\s+', value, re.IGNORECASE)
                        if m:
                            article = m.group(1)
                            value = value[m.end():]
                    if article and slot in _ARTICLE_SLOTS:
                        placeholder_pos = result.find(placeholder)
                        if not _should_keep_article(result[:placeholder_pos]):
                            skip_article = True
                        else:
                            skip_article = False
                            after_pos = placeholder_pos + len(placeholder)
                            if after_pos < len(result) and result[after_pos:].startswith("{suf"):
                                skip_article = True
                        if not skip_article:
                            value = article + " " + value
                    elif noun_form:
                        placeholder_pos = result.find(placeholder)
                        after_pos = placeholder_pos + len(placeholder)
                        if after_pos < len(result) and result[after_pos:].startswith(" of"):
                            value = noun_form
                    replacement = value
                    tr = {"slot": slot, "value": value, "source": src}
                    if _parent:
                        tr["parent"] = _parent
                    trace.append(tr)
                else:
                    replacement = str(entry)
                    tr = {"slot": slot, "value": replacement, "source": src}
                    if _parent:
                        tr["parent"] = _parent
                    trace.append(tr)
                result = result.replace(placeholder, replacement, 1)
                replaced = True
        if not replaced:
            break

    stripped = re.sub(r'^([Tt]he)\s(\S+)$', r'\2', result, 1)
    if stripped != result:
        result = stripped

    return result, trace


def expand_all(
    pattern: str,
    grammar: dict,
    depth: int = 0,
    max_depth: int = 10,
) -> list[str]:
    """Enumerate every expansion of a template."""
    if depth >= max_depth:
        return [pattern]

    ordered_slots = sorted(
        grammar.keys(),
        key=lambda k: (2 if k in _SUF_LAST else 0 if k in _PLACE_FIRST else 1),
    )
    for slot in ordered_slots:
        placeholder = "{" + slot + "}"
        if placeholder in pattern:
            entries = grammar.get(slot, [])
            if not entries:
                return [pattern]
            results = []
            for entry in entries:
                if isinstance(entry, dict) and "patterns" in entry:
                    for sub_pattern in entry["patterns"]:
                        for replacement in expand_all(
                            sub_pattern, grammar, depth + 1, max_depth
                        ):
                            new_pattern = pattern.replace(placeholder, replacement, 1)
                            results.extend(
                                expand_all(new_pattern, grammar, depth + 1, max_depth)
                            )
                elif isinstance(entry, dict) and "value" in entry:
                    value = entry["value"]
                    article = entry.get("article", "")
                    noun_form = entry.get("noun_form", "")
                    if not article and slot in _ARTICLE_SLOTS and value:
                        m = re.match(r'^(the|an?)\s+', value, re.IGNORECASE)
                        if m:
                            article = m.group(1)
                            value = value[m.end():]
                    if article and slot in _ARTICLE_SLOTS:
                        placeholder_pos = pattern.find(placeholder)
                        if not _should_keep_article(pattern[:placeholder_pos]):
                            skip_article = True
                        else:
                            skip_article = False
                            after_pos = placeholder_pos + len(placeholder)
                            if after_pos < len(pattern) and pattern[after_pos:].startswith("{suf"):
                                skip_article = True
                        if not skip_article:
                            value = article + " " + value
                    elif noun_form:
                        placeholder_pos = pattern.find(placeholder)
                        after_pos = placeholder_pos + len(placeholder)
                        if after_pos < len(pattern) and pattern[after_pos:].startswith(" of"):
                            value = noun_form
                    new_pattern = pattern.replace(placeholder, value, 1)
                    results.extend(expand_all(new_pattern, grammar, depth + 1, max_depth))
                else:
                    new_pattern = pattern.replace(placeholder, str(entry), 1)
                    results.extend(expand_all(new_pattern, grammar, depth + 1, max_depth))
            return results

    return [pattern]


def expand_all_traced(
    pattern: str,
    grammar: dict,
    depth: int = 0,
    max_depth: int = 10,
    _parent: str = "",
) -> list[tuple[str, list[dict]]]:
    """Enumerate every expansion of a template with provenance traces."""
    if depth >= max_depth:
        return [(pattern, [])]

    ordered_slots = sorted(
        grammar.keys(),
        key=lambda k: (2 if k in _SUF_LAST else 0 if k in _PLACE_FIRST else 1),
    )
    for slot in ordered_slots:
        placeholder = "{" + slot + "}"
        if placeholder in pattern:
            entries = grammar.get(slot, [])
            if not entries:
                return [(pattern, [])]
            results: list[tuple[str, list[dict]]] = []
            for entry in entries:
                src = entry.get("_source", "?")
                if isinstance(entry, dict) and "patterns" in entry:
                    for sub_pattern in entry["patterns"]:
                        for repl_text, sub_trace in expand_all_traced(
                            sub_pattern, grammar, depth + 1, max_depth,
                            _parent=slot,
                        ):
                            new_pattern = pattern.replace(placeholder, repl_text, 1)
                            for result_str, deeper in expand_all_traced(
                                new_pattern, grammar, depth + 1, max_depth,
                            ):
                                results.append((result_str, sub_trace + deeper))
                elif isinstance(entry, dict) and "value" in entry:
                    value = entry["value"]
                    article = entry.get("article", "")
                    noun_form = entry.get("noun_form", "")
                    if not article and slot in _ARTICLE_SLOTS and value:
                        m = re.match(r'^(the|an?)\s+', value, re.IGNORECASE)
                        if m:
                            article = m.group(1)
                            value = value[m.end():]
                    if article and slot in _ARTICLE_SLOTS:
                        placeholder_pos = pattern.find(placeholder)
                        if not _should_keep_article(pattern[:placeholder_pos]):
                            skip_article = True
                        else:
                            skip_article = False
                            after_pos = placeholder_pos + len(placeholder)
                            if after_pos < len(pattern) and pattern[after_pos:].startswith("{suf"):
                                skip_article = True
                        if not skip_article:
                            value = article + " " + value
                    elif noun_form:
                        placeholder_pos = pattern.find(placeholder)
                        after_pos = placeholder_pos + len(placeholder)
                        if after_pos < len(pattern) and pattern[after_pos:].startswith(" of"):
                            value = noun_form
                    tr = {"slot": slot, "value": value, "source": src}
                    if _parent:
                        tr["parent"] = _parent
                    new_pattern = pattern.replace(placeholder, value, 1)
                    for result_str, deeper in expand_all_traced(
                        new_pattern, grammar, depth + 1, max_depth,
                    ):
                        results.append((result_str, [tr] + deeper))
                else:
                    tr = {"slot": slot, "value": str(entry), "source": src}
                    if _parent:
                        tr["parent"] = _parent
                    new_pattern = pattern.replace(placeholder, str(entry), 1)
                    for result_str, deeper in expand_all_traced(
                        new_pattern, grammar, depth + 1, max_depth,
                    ):
                        results.append((result_str, [tr] + deeper))
            return results

    return [(pattern, [])]


def title_case(text: str) -> str:
    """Title-case a generated title, preserving known acronyms and small words."""
    small = {"of", "the", "in", "and", "for", "a", "an", "from", "over", "upon", "against", "beyond", "yet", "or", "without", "near", "under", "within", "through", "after", "before", "between", "among", "versus", "by"}
    preserve = {
        # SI unit symbols
        "m",
        "g",
        "s",
        "A",
        "K",
        "mol",
        "cd",
        "N",
        "J",
        "W",
        "Pa",
        "Hz",
        "V",
        "C",
        "Ω",
        # Group theory symbols
        "Z",
        "S",
        "D",
        "A",
        "GL",
        "SL",
        "O",
        "SO",
        # Corporate acronyms
        "CEO",
        "CFO",
        "CTO",
        "COO",
        "VP",
        "KPIs",
        "OKRs",
        # Tech/buzzword acronyms
        "AI",
        "NFT",
        "ROI",
    }

    words = text.split()
    if not words:
        return text
    result = []
    for i, word in enumerate(words):
        if word in preserve:
            result.append(word)
        elif word.upper() in preserve:
            result.append(word.upper())
        elif i == 0 or word.lower() not in small:
            result.append(word[:1].upper() + word[1:])
        else:
            result.append(word)
    return " ".join(result)


def _resolve_level(
    level: str | int | None,
) -> tuple[str | None, bool]:
    """Return (level_name, exact)."""
    if level is None:
        return None, False
    if isinstance(level, int):
        if 0 <= level < len(LEVELS):
            return LEVELS[level], False
        raise ValueError(f"Level index {level} out of range (0-{len(LEVELS) - 1}).")
    raw = str(level).strip()
    exact = False
    if raw.endswith("!"):
        exact = True
        raw = raw[:-1]
    if raw.isdigit():
        idx = int(raw)
        if 0 <= idx < len(LEVELS):
            return LEVELS[idx], exact
        raise ValueError(f"Level index {idx} out of range (0-{len(LEVELS) - 1}).")
    if raw == "all":
        return None, False
    if raw in LEVELS:
        return raw, exact
    raise ValueError(f"Unknown level: {raw}. Use {', '.join(LEVELS)} or 0-3.")


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


def build_grammar(
    selected_flavors: list[str],
    slot_sources: dict[str, list[str]] | dict[str, list[list[str]]],
    primary: Path,
    fallback: Path | None,
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

    source_names = ["generic", "numbers"] + selected_flavors
    grammars: dict[str, dict] = {}
    for name in source_names:
        grammars[name] = _apply_expose(_load_toml(name, primary, fallback))

    flat: dict[str, list] = {}
    for g in grammars.values():
        for key, value in g.items():
            if key in ("expose", "category"):
                continue
            if isinstance(value, list):
                flat.setdefault(key, []).extend(value)

    generic_slots: set[str] = set()
    for g in grammars.values():
        for entry in g.get("expose", []):
            generic_slots.add(entry["slot"])

    all_slots: set[str] = set()
    for g in grammars.values():
        all_slots.update(g.keys())
    all_slots.discard("expose")
    all_slots.discard("category")

    merged: dict[str, list] = {}
    for slot in all_slots:
        entries: list[dict] = []
        if slot in generic_slots:
            groups = _slot_groups.get(slot)
            if groups is None:
                flavor_sources = [
                    s for s in source_names if s not in ("generic", "numbers")
                ]
                has_flavor = any(
                    slot in grammars[s] and grammars[s][slot] for s in flavor_sources
                )
                groups = [flavor_sources if has_flavor else source_names]
            else:
                # Filter out generic/numbers from each group
                groups = [
                    [s for s in grp if s not in ("generic", "numbers")]
                    for grp in groups
                ]

            for grp in groups:
                for source in grp:
                    if source.endswith("!"):
                        entries.append(
                            {
                                "value": source[:-1],
                                "prefixable": False,
                                "suffixible": False,
                                "_source": "literal",
                            }
                        )
                    elif source.startswith("@"):
                        # @flavor or @flavor:group
                        rest = source[1:]
                        if ":" in rest:
                            fname, group = rest.split(":", 1)
                            if fname in grammars and group in grammars[fname]:
                                tagged = [dict(e, _source=source) for e in grammars[fname][group]]
                                entries.extend(tagged)
                            else:
                                raise ValueError(
                                    f"Unknown @reference '{source}' for slot '{slot}'."
                                )
                        else:
                            if rest in grammars and slot in grammars[rest]:
                                tagged = [dict(e, _source=source) for e in grammars[rest][slot]]
                                entries.extend(tagged)
                            else:
                                raise ValueError(
                                    f"Unknown @flavor '{rest}' for slot '{slot}'."
                                )
                    elif source.endswith(":"):
                        ikey = source[:-1]
                        if ikey in flat:
                            tagged = [dict(e, _source=source.rstrip(":")) for e in flat[ikey]]
                            entries.extend(tagged)
                        else:
                            raise ValueError(
                                f"Unknown internal group '{ikey}' for slot '{slot}'."
                            )
                    elif source in grammars:
                        if slot in grammars[source]:
                            tagged = [dict(e, _source=source) for e in grammars[source][slot]]
                            entries.extend(tagged)
                            # If a merged internal group with the same name exists,
                            # include it too (it spans all flavours). Delivery
                            # groups (melee/ranged/area/status) only contribute to
                            # the "technique" slot.
                            if source in flat and slot == "technique":
                                tagged2 = [dict(e, _source=":" + source) for e in flat[source]]
                                entries.extend(tagged2)
                    elif source in flat:
                        tagged = [dict(e, _source=source) for e in flat[source]]
                        entries.extend(tagged)
                    else:
                        valid = [s for s in source_names if s in grammars and slot in grammars[s] and grammars[s][slot]]
                        hint = ""
                        if valid:
                            hint = f" Available: {', '.join(valid)}"
                        raise ValueError(
                            f"Unknown source '{source}' for slot '{slot}'.{hint}"
                        )
            # Fallback: if restricted sources yielded nothing, use all sources
            if not entries and _slot_groups.get(slot):
                for source in source_names:
                    if source in grammars and slot in grammars[source]:
                        tagged = [dict(e, _source=source) for e in grammars[source][slot]]
                        entries.extend(tagged)
        else:
            for source in source_names:
                if source in grammars and slot in grammars[source]:
                    tagged = [dict(e, _source=source) for e in grammars[source][slot]]
                    entries.extend(tagged)
        if entries:
            merged[slot] = entries

    return merged


def load_title_grammar(
    story: str | None = None,
    config: AraSettings | None = None,
    flavors: list[str] | str | None = None,
    slot_sources: dict[str, list[str]] | None = None,
) -> dict:
    """Load and merge title flavor grammars.

    :param flavors: Flavor name(s) to load. ``None`` loads every available flavor.
    :param slot_sources: Optional per-slot source restrictions.
    """
    primary, fallback = _title_dirs(story, config)
    available = list_title_flavors(story, config)

    if flavors is None:
        selected = available
    elif isinstance(flavors, str):
        selected = [flavors]
    else:
        selected = list(flavors)

    unknown = [s for s in selected if s not in available]
    if unknown:
        raise ValueError(f"Unknown title flavor(s): {', '.join(unknown)}")

    return build_grammar(
        selected,
        slot_sources or {},
        primary,
        fallback,
    )


def generate_title(
    story: str | None = None,
    config: AraSettings | None = None,
    template: str | None = None,
    flavors: list[str] | str | None = None,
    level: str | int | None = "2",
    slot_sources: dict[str, list[str]] | None = None,
    required_slots: list[str] | set[str] | None = None,
) -> str:
    """Generate a random title.

    :param template: Specific template string or ``None`` for random.
    :param flavors: Flavor name(s) to use, or ``None`` for all.
    :param level: Complexity level name/index, optionally suffixed with ``!`` for exact.
    :param slot_sources: Per-slot source restrictions.
    :param required_slots: Only use templates containing all of these slots.
    """
    primary, fallback = _title_dirs(story, config)
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

    grammar = load_title_grammar(
        story=story,
        config=config,
        flavors=flavors,
        slot_sources=slot_sources,
    )

    if template is not None:
        tmpl = template
    else:
        tmpl = random.choice(templates)
    return title_case(expand(tmpl, grammar))
