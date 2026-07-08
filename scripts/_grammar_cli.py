"""Shared CLI logic for title_generate.py and ability_generate.py."""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable


def parse_level(level: str | None) -> tuple[str | None, bool]:
    from ara.world.title import _resolve_level
    return _resolve_level(level)


def build_parser(
    name: str,
    available_flavors: list[str],
    default_level: str,
    slot_examples: str = "",
    epilog: str = "",
) -> argparse.ArgumentParser:
    plural = f"{name[:-1]}ies" if name.endswith("y") else f"{name}s"
    parser = argparse.ArgumentParser(
        description=f'Generate {plural} from composable flavor slot pools.',
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'flavor_or_count',
        nargs='?',
        default='all',
        help='Flavor name(s) to use (comma-separated for multiple), or a number for the count (default: all).',
    )
    parser.add_argument('count', nargs='?', type=int, help='Number of samples (optional).')
    parser.add_argument('--story', default=None, help=f'Story ID for per-story {name} overrides.')
    parser.add_argument(
        '-t', '--template',
        default=None,
        help='Use a specific template string or integer index instead of picking randomly.',
    )
    parser.add_argument(
        '-s', '--slot', dest='slot_sources', action='append', nargs='*',
        metavar='SLOT[:MOD] [SOURCES]',
        help='Restrict a slot to a list of sources. '
        'SLOT may include :modifiers (e.g. noun:plural, ordinal:>1,<6). '
        'Sources are comma or plus separated. '
        'A bare source that matches a merged technique/delivery group '
        '(e.g. status, melee, area for abilities) pulls from all flavors '
        'that define that group. '
        'Use @flavor to explicitly pull this slot from a specific flavor '
        '(stricter than a bare flavor name; it must expose the slot). '
        'Use @flavor:group to pull a specific internal group from that flavor. '
        'Append ! for a literal value. '
        'Give just the slot name to see available sources, '
        'or give no arguments to list all slots. '
        + (f'Example: {slot_examples}' if slot_examples else ''),
    )
    parser.add_argument(
        '-f', '--flavor', dest='flavors', action='append', nargs='?', const='',
        metavar='FLAVOR',
        help='Flavor name(s) to use (can be specified multiple times). Overrides positional flavor selection. '
        'Use -f without arguments to list available flavors.',
    )
    parser.add_argument(
        '-l', '--level', default=default_level,
        help=f'Complexity level. Append "!" for that level only. (default: {default_level}).',
    )
    parser.add_argument('-L', '--list', dest='list_templates', action='store_true',
                        help='List all available templates with integer indices and exit.')
    parser.add_argument('-i', '--inspect', nargs='?', const='__all__', default=None,
                        help='Show a summary of the loaded grammar and exit. '
                        'Optionally specify a slot name to inspect only that slot.')
    parser.add_argument('-a', '--all', dest='enumerate_all', action='store_true',
                        help='Enumerate every expansion instead of sampling randomly. '
                        'Combine with --inspect to dump full slot contents.')
    parser.add_argument('--all-force-destructive-i-know-what-im-doing', action='store_true',
                        dest='all_force_destructive',
                        help='Allow --all with no restricting flags. This can produce enormous output and heavy CPU/memory use; only use if you know what you are doing.')
    parser.add_argument('-r', '--require', dest='required_slots', action='append', nargs='?',
                        metavar='SLOTS', const='',
                        help='Only use templates that contain these slots. '
                        'Multiple slots comma-separated = ALL required (AND). '
                        'Plus-separated = ANY required (OR). '
                        'Example: -r "noun,verb" (both), -r "noun+verb" (either). '
                        'A bare name matches any form, e.g. -r verb also matches {verb:noun}. '
                        'Prefix with @ to require the exact bare slot, e.g. -r @verb matches only {verb}. '
                        'Append :source... to also restrict that slot, e.g. '
                        '-r verb:melee is shorthand for -r verb -s verb melee, '
                        'and -r verb:status restricts verb to the cross-flavor '
                        'status group. '
                        'Use -r without arguments to list available slots.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Per-sample: show template and slot provenance trace.')
    return parser


def parse_flavor_arg(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    available_flavors: list[str],
    categorized: dict[str, list[str]] | None = None,
) -> tuple[str, int]:
    flavor_arg = 'all'
    count = args.count if args.count is not None else 10
    if args.flavor_or_count and args.flavor_or_count.isdigit():
        count = int(args.flavor_or_count)
        flavor_arg = 'all'
    elif args.flavor_or_count and args.flavor_or_count != 'all':
        names = [f.strip() for f in args.flavor_or_count.split(',')]
        unknown = [f for f in names if f not in available_flavors]
        if unknown:
            parser.error(
                f'Unknown flavor(s): {", ".join(unknown)}. '
                f'Available: {", ".join(available_flavors)}'
            )
        flavor_arg = ','.join(names)
    if args.flavors:
        if any(f == '' or f is None for f in args.flavors):
            if categorized:
                print('Available flavors:')
                for cat in sorted(categorized):
                    print(f'  {cat}: {", ".join(sorted(categorized[cat]))}')
            else:
                print(f'Available flavors: {", ".join(available_flavors)}')
            sys.exit(0)
        flavor_arg = ','.join(args.flavors)
    return flavor_arg, count


def parse_slot_args(
    raw_slots: list[list[str]] | None,
) -> tuple[dict[str, list[list[str]]], dict[str, list[str]], list[str], bool]:
    """Return (slot_sources, slot_modifiers, slot_queries, list_all).

    slot_sources maps slot_name -> list of groups (groups are unioned).
    slot_modifiers maps slot_name -> list of modifiers parsed from the slot
    argument, e.g. ``--slot noun:plural`` gives ``{"noun": ["plural"]}``.

    ``,`` and ``+`` are both separators; within a group all sources
    contribute (union).
    """
    slot_sources: dict[str, list[list[str]]] = {}
    slot_modifiers: dict[str, list[str]] = {}
    slot_queries: list[str] = []
    list_all = False
    if raw_slots:
        for item in raw_slots:
            if len(item) == 0:
                list_all = True
                continue
            # The first argument may contain modifiers: noun:plural:>1
            parts = item[0].split(":")
            slot = parts[0].strip()
            modifiers = [p.strip() for p in parts[1:] if p.strip()]
            if modifiers:
                slot_modifiers[slot] = modifiers
            if len(item) == 1:
                slot_queries.append(slot)
            elif len(item) >= 2:
                # , and + both split sources; everything in one group
                sources: list[str] = []
                for token in re.split(r'[,+]', item[1]):
                    s = token.strip()
                    if s:
                        sources.append(s)
                slot_sources[slot] = [sources]
    return slot_sources, slot_modifiers, slot_queries, list_all


def parse_require_args(
    required_raw: list[str] | None,
) -> tuple[list[str], dict[str, list[list[str]]]]:
    """Parse ``-r`` expressions into cleaned require strings and slot sources.

    A slot name may be followed by ``:source...`` to restrict that slot for
    this run. For example, ``-r verb:melee`` is equivalent to
    ``-r verb -s verb melee``.

    ``-r foo:@bar:baz+a:b`` becomes:

    * require string: ``foo+a``
    * slot sources: ``foo -> [["@bar:baz"]], a -> [["b"]]``

    An ``@`` prefix on the slot itself marks an exact bare reference;
    sources may still follow, e.g. ``-r @verb:melee`` requires a bare
    ``{verb}`` and restricts the verb slot to melee.
    """
    if not required_raw:
        return [], {}
    cleaned: list[str] = []
    slot_sources: dict[str, list[list[str]]] = {}
    for expr in required_raw:
        if not expr or not isinstance(expr, str):
            continue
        groups = expr.split(',')
        cleaned_groups: list[str] = []
        for gs in groups:
            members = gs.split('+')
            cleaned_members: list[str] = []
            for m in members:
                m = m.strip()
                if not m:
                    continue
                exact = m.startswith('@')
                if exact:
                    m = m[1:]
                slot, _, rest = m.partition(':')
                slot = slot.strip()
                if rest:
                    source = rest.strip()
                    if source:
                        slot_sources.setdefault(slot, []).append([source])
                cleaned_members.append(('@' if exact else '') + slot)
            cleaned_groups.append('+'.join(cleaned_members))
        cleaned.append(','.join(cleaned_groups))
    return cleaned, slot_sources


def handle_slot_queries(
    slot_queries: list[str],
    selected_flavors: list[str],
    dirs_func: Callable[[str | None], tuple[Path, Path | None]],
    always_load: frozenset[str],
    story: str | None = None,
) -> None:
    """Print available sources for queried slots and exit."""
    from ara.world.title import _load_toml, _apply_expose
    primary, fallback = dirs_func(story)
    source_names = ["generic", "numbers"] + selected_flavors

    # Collect category info per source
    categories: dict[str, str] = {}
    for s in source_names:
        try:
            g = _load_toml(s, primary, fallback)
            cat = g.get("category", "base" if s in ("generic", "numbers") else "other")
            categories[s] = cat
        except FileNotFoundError:
            categories[s] = "other"

    for slot in slot_queries:
        by_cat: dict[str, list[str]] = {}
        for s in source_names:
            try:
                g = _load_toml(s, primary, fallback)
                g = _apply_expose(g)
                if slot in g and g[slot]:
                    cat = categories.get(s, "other")
                    by_cat.setdefault(cat, []).append(s)
            except FileNotFoundError:
                pass
        if by_cat:
            print(f'Slot [{slot}]:')
            for cat in sorted(by_cat):
                print(f'  {cat}: {", ".join(sorted(by_cat[cat]))}')
        else:
            print(f'Slot [{slot}]: (no sources)')
    sys.exit(1)


def load_templates_raw(
    dirs_func: Callable[[str | None], tuple[Path, Path | None]],
    story: str | None,
    level_name: str | None,
    exact: bool,
) -> list[tuple[str, str]]:
    """Return a list of (template, level) tuples."""
    from ara.world.title import _load_toml
    primary, fallback = dirs_func(story)
    td = _load_toml("templates", primary, fallback)
    levels = ["simple", "moderate", "complex", "insane"]
    if level_name:
        if exact:
            return [(tmpl, level_name) for tmpl in td.get(level_name, [])]
        templates: list[tuple[str, str]] = []
        for lvl in levels:
            for tmpl in td.get(lvl, []):
                templates.append((tmpl, lvl))
            if lvl == level_name:
                break
        return templates
    templates = []
    for lvl in levels:
        for tmpl in td.get(lvl, []):
            templates.append((tmpl, lvl))
    return templates


def _tmpl_text(t: str | tuple[str, ...]) -> str:
    """Return the template string from a plain string or a (template, ...) tuple."""
    return t[0] if isinstance(t, tuple) else t


def _slot_base(ref: str) -> str:
    """Return the base slot name from a template reference like ``verb:noun``.

    ``{prefix+}`` is treated as ``prefix`` so that ``-r prefix`` also matches
    prefix-stack references.
    """
    base = ref.split(':', 1)[0]
    return base.rstrip('+')


def filter_templates(
    templates: list[str],
    required_raw: list | None,
) -> list[str]:
    """Filter templates by required slots.

    *required_raw* is a list of lists from ``-r`` flags with ``nargs='*'``.
    Each inner list contains slot expression strings.
    Each expression is comma-separated AND groups with plus-separated
    OR members within a group.

    A bare slot name matches any reference form, so ``-r verb`` matches both
    ``{verb}`` and ``{verb:noun}``. Prefix with ``@`` to require the exact
    bare reference, e.g. ``-r @verb`` matches ``{verb}`` only.

    ``-r noun,verb`` → both noun AND verb required
    ``-r noun+verb`` → noun OR verb required
    """
    if not required_raw:
        return templates
    # -r with nargs='?' appends None for bare -r; filter those out
    expressions: list[str] = [e for e in required_raw if e and e is not None and isinstance(e, str)]
    if not expressions:
        return templates
    # Parse: each expression → AND groups (comma), each group = OR members (plus)
    and_groups: list[list[tuple[str, bool]]] = []
    for expr in expressions:
        group_strs = expr.split(',')
        for gs in group_strs:
            members: list[tuple[str, bool]] = []
            for s in gs.split('+'):
                s = s.strip()
                if s.startswith('@'):
                    members.append((s[1:], True))  # exact bare reference
                else:
                    members.append((s, False))  # base-name match
            and_groups.append(members)
    result = []
    for t in templates:
        text = _tmpl_text(t)
        t_refs = set(re.findall(r'\{([^}]+)\}', text))
        t_bases = set(_slot_base(ref) for ref in t_refs)
        ok = True
        for grp in and_groups:
            if not any(
                (exact and req in t_refs) or (not exact and req in t_bases)
                for req, exact in grp
            ):
                ok = False
                break
        if ok:
            result.append(t)
    if not result:
        flat = sorted(set(req for grp in and_groups for req, _ in grp))
        print(f'No templates match required slots: {", ".join(flat)}')
        sys.exit(1)
    return result


def list_templates(dirs_func, story):
    from ara.world.title import _load_toml
    primary, fallback = dirs_func(story)
    td = _load_toml("templates", primary, fallback)
    levels = ["simple", "moderate", "complex", "insane"]
    print('Available templates:\n')
    template_list = []
    for lvl in levels:
        for tmpl in td.get(lvl, []):
            template_list.append((tmpl, lvl))
    for i, (tmpl, lvl) in enumerate(template_list):
        print(f'  {i:3d} [{lvl}]  {tmpl}')
    print(f'\nTotal: {len(template_list)}')


def resolve_template_idx(
    template: str | None,
    dirs_func,
    story: str | None,
) -> tuple[str, str | None] | None:
    """Resolve a template argument.

    Returns the argument unchanged as ``(template, None)`` if it is not an
    integer index, or ``(template, level)`` for the indexed template.
    """
    if template is None:
        return None
    if not template.lstrip('-').isdigit():
        return (template, None)
    from ara.world.title import _load_toml
    primary, fallback = dirs_func(story)
    td = _load_toml("templates", primary, fallback)
    flat: list[tuple[str, str]] = []
    for lvl in ["simple", "moderate", "complex", "insane"]:
        for tmpl in td.get(lvl, []):
            flat.append((tmpl, lvl))
    idx = int(template)
    if 0 <= idx < len(flat):
        return flat[idx]
    print(f'Template index {idx} out of range (0-{len(flat) - 1}).')
    sys.exit(1)


def _display_entry(entry) -> str:
    if isinstance(entry, dict):
        if 'value' in entry:
            return entry['value']
        if 'patterns' in entry:
            return ' \u21e2 '.join(entry['patterns'])
    return str(entry)


def inspect_grammar(
    grammar: dict,
    selected_flavors: list[str],
    templates: list[str],
    flatten: set[str],
    all_entries: bool = False,
    slot_filter: str | None = None,
) -> None:
    print('Loaded flavors:', ', '.join(selected_flavors) if selected_flavors else 'none')
    print(f'Templates available: {len(templates)}')
    print()

    if slot_filter and slot_filter not in flatten and slot_filter not in grammar:
        print(f'Slot "{slot_filter}" not found in loaded grammar.')
        print(f'Available slots: {", ".join(sorted(flatten))}')
        return

    print('Generic slot sizes:')
    if slot_filter:
        count = len(grammar.get(slot_filter, []))
        print(f'  {slot_filter:15s} {count:4d} entries')
    else:
        for slot in sorted(flatten):
            count = len(grammar.get(slot, []))
            print(f'  {slot:15s} {count:4d} entries')
    print()

    internal = [k for k in grammar.keys() if k not in flatten and k != 'expose']

    pattern_keys, mixed_keys, value_keys = set(), set(), set()
    for k in internal:
        entries = grammar.get(k, [])
        if not entries:
            continue
        has_pattern = any(isinstance(e, dict) and 'patterns' in e for e in entries)
        has_value = any(isinstance(e, dict) and 'value' in e for e in entries)
        if has_pattern and has_value:
            mixed_keys.add(k)
        elif has_pattern:
            pattern_keys.add(k)
        elif has_value:
            value_keys.add(k)
        else:
            value_keys.add(k)

    children_of: dict[str, list[str]] = {}
    pattern_parents = set(pattern_keys | mixed_keys)
    for slot in flatten:
        entries = grammar.get(slot, [])
        if any(isinstance(e, dict) and 'patterns' in e for e in entries):
            pattern_parents.add(slot)
    for k in pattern_parents:
        children_of[k] = []
        for e in grammar[k]:
            for pat in e.get('patterns', []):
                for ref in set(re.findall(r'\{([^}+]+)\}', pat)):
                    if ref in internal:
                        children_of[k].append(ref)

    referenced = set()
    for refs in children_of.values():
        referenced.update(refs)
    pattern_roots = [k for k in pattern_keys if k not in referenced]
    mixed_roots = [k for k in mixed_keys if k not in referenced]
    generic_roots = [k for k in flatten if k in pattern_parents and k not in referenced]
    roots = pattern_roots + mixed_roots + generic_roots
    standalone_values = [k for k in value_keys if k not in referenced]

    def _resolve_count(key: str, seen: set[str] | None = None) -> int:
        if seen is None:
            seen = set()
        if key in seen:
            return 0
        seen = seen | {key}
        entries = grammar.get(key, [])
        if not entries:
            return 0
        if all(isinstance(e, dict) and 'value' in e for e in entries):
            return len(entries)
        total = 0
        for e in entries:
            if isinstance(e, dict) and 'patterns' in e:
                for pat in e['patterns']:
                    for ref in set(re.findall(r'\{([^}+]+)\}', pat)):
                        if ref in internal:
                            total += _resolve_count(ref, seen)
            elif isinstance(e, dict) and 'value' in e:
                total += 1
            else:
                total += 1
        return total

    def _collect_descendants(key: str, children: dict[str, list[str]]) -> set[str]:
        """Return all descendant keys of *key* in the children graph."""
        result: set[str] = set()
        for child in children.get(key, []):
            result.add(child)
            result.update(_collect_descendants(child, children))
        return result

    def _is_internal(key: str) -> bool:
        return key.startswith('_')

    def _print_tree(key: str, depth: int = 0, seen: set[str] | None = None) -> None:
        if seen is None:
            seen = set()
        if key in seen:
            print(f'{"  " * depth}{key:30s} (cycle)')
            return
        seen = seen | {key}
        entries = grammar.get(key, [])
        suffix = ' (internal)' if _is_internal(key) else ''
        if not entries:
            print(f'{"  " * depth}{key:30s} 0 entries{suffix}')
            return
        has_pattern = any(isinstance(e, dict) and 'patterns' in e for e in entries)
        if has_pattern and key not in flatten:
            print(f'{"  " * depth}{key:30s} {_resolve_count(key):4d} entries (lazy){suffix}')
        else:
            print(f'{"  " * depth}{key:30s} {len(entries):4d} entries{suffix}')
        for child in sorted(set(children_of.get(key, [])), key=lambda c: (_is_internal(c), c)):
            _print_tree(child, depth + 1, seen)

    def _root_sort(key: str):
        return (_is_internal(key), key)

    if slot_filter:
        # Find roots that contain or are ancestors of the filtered slot
        filtered_roots: list[str] = []
        if slot_filter in flatten:
            filtered_roots.append(slot_filter)
        # Also include any internal groups that are ancestors or the slot itself
        for root in sorted(roots + standalone_values, key=_root_sort):
            if root == slot_filter:
                filtered_roots.append(root)
            elif slot_filter in _collect_descendants(root, children_of):
                filtered_roots.append(root)
        _print_filtered = sorted(set(filtered_roots), key=_root_sort)
    else:
        _print_filtered = sorted(roots, key=_root_sort) + sorted(standalone_values, key=_root_sort)

    print(f'Internal groups/patterns: {len(internal)}')
    for root in _print_filtered:
        _print_tree(root)
    print()

    if all_entries:
        print('=== Full Slot Contents ===')
        _slots_to_show = [slot_filter] if slot_filter else sorted(flatten)
        for slot in _slots_to_show:
            entries = grammar.get(slot, [])
            if not entries:
                continue
            seen_sources: dict[str, list[str]] = {}
            for entry in entries:
                key = _display_entry(entry)
                src = entry.get('_source', '?')
                seen_sources.setdefault(key, []).append(src)
            total = sum(len(v) for v in seen_sources.values())
            unique = len(seen_sources)
            suffix = f' ({unique} unique)' if unique < total else ''
            print(f'\n[{slot}]  ({total} entries{suffix})')
            for value, sources in sorted(seen_sources.items()):
                deduped = sorted(set(sources))
                print(f'  {value:30s} [{", ".join(deduped)}]')
        _internal_to_show = _collect_descendants(slot_filter, children_of) | ({slot_filter} & set(internal)) if slot_filter else sorted(internal)
        for key in sorted(_internal_to_show, key=lambda k: (k.startswith('_'), k)):
            entries = grammar.get(key, [])
            if not entries:
                continue
            seen_sources = {}
            for entry in entries:
                k = _display_entry(entry)
                src = entry.get('_source', '?')
                seen_sources.setdefault(k, []).append(src)
            total = sum(len(v) for v in seen_sources.values())
            unique = len(seen_sources)
            suffix = f' ({unique} unique)' if unique < total else ''
            marker = ' (internal)' if key.startswith('_') else ''
            print(f'\n[{key}]{marker}  ({total} entries{suffix})')
            for value, sources in sorted(seen_sources.items()):
                deduped = sorted(set(sources))
                print(f'  {value:30s} [{", ".join(deduped)}]')
    else:
        print('Sample entries:')
        _sample_slots = [slot_filter] if slot_filter else sorted(flatten)
        for slot in _sample_slots:
            entries = grammar.get(slot, [])
            if not entries:
                continue
            samples = []
            for entry in entries[:5]:
                samples.append(_display_entry(entry))
            print(f'  {slot}: {", ".join(samples)}')


def print_verbose(tmpl: str, trace: list[dict], level: str | None = None) -> None:
    level_part = f' (level={level})' if level else ''
    print(f'    template: {tmpl}{level_part}')
    for t in trace:
        val = t['value'].strip()
        if val:
            slot = t['slot']
            mods = t.get('modifiers')
            if mods:
                slot += ':' + ':'.join(mods)
            slot_display = f'{t["parent"]} -> {slot}' if t.get('parent') else slot
            print(f'    - {slot_display} [{t["source"]}] -> {val}')
