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
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f'Sample {name}s from {name} flavor slot pools.',
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
        metavar='SLOT [SOURCES]',
        help='Restrict a slot to a comma-separated list of sources, '
        'give just the slot name to see available sources, '
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
    parser.add_argument('--all-force', action='store_true',
                        help='Allow --all with no restricting flags (may produce enormous output).')
    parser.add_argument('-r', '--require', dest='required_slots', action='append', nargs='?',
                        metavar='SLOTS', const='',
                        help='Only use templates that contain these slots. '
                        'Multiple slots comma-separated = ALL required (AND). '
                        'Plus-separated = ANY required (OR). '
                        'Example: -r "domain,technique" (both), -r "domain+technique" (either).'
                        ' Use -r without arguments to list available slots.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Per-sample: show template and slot provenance trace.')
    return parser


def parse_flavor_arg(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    available_flavors: list[str],
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
            print(f'Available flavors: {", ".join(available_flavors)}')
            sys.exit(0)
        flavor_arg = ','.join(args.flavors)
    return flavor_arg, count


def parse_slot_args(
    raw_slots: list[list[str]] | None,
) -> tuple[dict[str, list[list[str]]], list[str], bool]:
    """Return (slot_sources, slot_queries, list_all).

    slot_sources maps slot_name -> list of groups (groups are unioned).
    ``,`` and ``+`` are both separators; within a group all sources
    contribute (union).
    """
    slot_sources: dict[str, list[list[str]]] = {}
    slot_queries: list[str] = []
    list_all = False
    if raw_slots:
        for item in raw_slots:
            if len(item) == 0:
                list_all = True
            elif len(item) == 1:
                slot_queries.append(item[0])
            elif len(item) >= 2:
                # , and + both split sources; everything in one group
                sources: list[str] = []
                for token in re.split(r'[,+]', item[1]):
                    s = token.strip()
                    if s:
                        sources.append(s)
                slot_sources[item[0]] = [sources]
    return slot_sources, slot_queries, list_all


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
) -> list[str]:
    from ara.world.title import _load_toml
    primary, fallback = dirs_func(story)
    td = _load_toml("templates", primary, fallback)
    levels = ["simple", "moderate", "complex", "insane"]
    if level_name:
        if exact:
            return list(td.get(level_name, []))
        templates: list[str] = []
        for lvl in levels:
            templates.extend(td.get(lvl, []))
            if lvl == level_name:
                break
        return templates
    templates = []
    for lvl in levels:
        templates.extend(td.get(lvl, []))
    return templates


def filter_templates(
    templates: list[str],
    required_raw: list | None,
) -> list[str]:
    """Filter templates by required slots.

    *required_raw* is a list of lists from ``-r`` flags with ``nargs='*'``.
    Each inner list contains slot expression strings.
    Each expression is comma-separated AND groups with plus-separated
    OR members within a group.

    ``-r domain,technique`` → both domain AND technique required
    ``-r domain+technique`` → domain OR technique required
    """
    if not required_raw:
        return templates
    # -r with nargs='?' appends None for bare -r; filter those out
    expressions: list[str] = [e for e in required_raw if e and e is not None and isinstance(e, str)]
    if not expressions:
        return templates
    # Parse: each expression → AND groups (comma), each group = OR members (plus)
    and_groups: list[list[str]] = []
    for expr in expressions:
        group_strs = expr.split(',')
        or_members = [s.strip() for s in group_strs[0].split('+')]
        and_groups.append(or_members)
        for gs in group_strs[1:]:
            and_groups.append([s.strip() for s in gs.split('+')])
    result = []
    for t in templates:
        t_slots = set(re.findall(r'\{([^}+]+)\}', t))
        ok = True
        for grp in and_groups:
            if not any(s in t_slots for s in grp):
                ok = False
                break
        if ok:
            result.append(t)
    if not result:
        flat = sorted(set(s for grp in and_groups for s in grp))
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
) -> str | None:
    if template is None or not template.lstrip('-').isdigit():
        return template
    from ara.world.title import _load_toml
    primary, fallback = dirs_func(story)
    td = _load_toml("templates", primary, fallback)
    flat: list[str] = []
    for lvl in ["simple", "moderate", "complex", "insane"]:
        flat.extend(td.get(lvl, []))
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

    def _print_tree(key: str, depth: int = 0, seen: set[str] | None = None) -> None:
        if seen is None:
            seen = set()
        if key in seen:
            print(f'{"  " * depth}{key:30s} (cycle)')
            return
        seen = seen | {key}
        entries = grammar.get(key, [])
        if not entries:
            print(f'{"  " * depth}{key:30s} 0 entries')
            return
        has_pattern = any(isinstance(e, dict) and 'patterns' in e for e in entries)
        if has_pattern and key not in flatten:
            print(f'{"  " * depth}{key:30s} {_resolve_count(key):4d} entries (lazy)')
        else:
            print(f'{"  " * depth}{key:30s} {len(entries):4d} entries')
        for child in sorted(set(children_of.get(key, []))):
            _print_tree(child, depth + 1, seen)

    if slot_filter:
        # Find roots that contain or are ancestors of the filtered slot
        filtered_roots: list[str] = []
        if slot_filter in flatten:
            filtered_roots.append(slot_filter)
        # Also include any internal groups that are ancestors or the slot itself
        for root in sorted(roots + standalone_values):
            if root == slot_filter:
                filtered_roots.append(root)
            elif slot_filter in _collect_descendants(root, children_of):
                filtered_roots.append(root)
        _print_filtered = sorted(set(filtered_roots))
    else:
        _print_filtered = sorted(roots) + sorted(standalone_values)

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
        for key in sorted(_internal_to_show):
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
            print(f'\n[{key}]  ({total} entries{suffix})')
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


def print_verbose(tmpl: str, trace: list[dict]) -> None:
    print(f'    template: {tmpl}')
    for t in trace:
        val = t['value'].strip()
        if val:
            slot_display = f'{t["parent"]} -> {t["slot"]}' if t.get('parent') else t['slot']
            print(f'    - {slot_display} [{t["source"]}] -> {val}')
