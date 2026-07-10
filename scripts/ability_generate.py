#!/usr/bin/env python3
"""CLI for sampling abilities from the Ara ability grammar."""

from __future__ import annotations

import random
import re
import sys
from typing import Any

from _grammar_cli import (
    build_parser,
    filter_templates,
    handle_slot_queries,
    inspect_grammar,
    list_templates,
    load_templates_raw,
    parse_flavor_arg,
    parse_level,
    parse_require_args,
    parse_slot_args,
    print_verbose,
    resolve_template_idx,
)

from ara.world import ability
from ara.world.title import (
    cull_grammar,
    expand_traced,
    expand,
    expand_all,
    expand_all_traced,
)

STORY = "ability"
DEFAULT_LEVEL = "complex"
SLOTS = ability._SLOTS
ALWAYS_LOAD = frozenset(ability._ALWAYS_LOAD)
_SLOT_EXAMPLE = "--slot noun fire --slot verb:noun @melee --slot verb status"


def _apply_modifiers(
    templates: list[str] | list[tuple[str, str]],
    modifiers: dict[str, list[str]],
) -> list[str] | list[tuple[str, str]]:
    """Rewrite slot placeholders to include CLI modifiers."""
    if not modifiers:
        return templates
    result = list(templates)
    for slot, mods in modifiers.items():
        suffix = ":" + ":".join(mods)
        pattern = re.compile(rf"\{{{re.escape(slot)}(?!:)\}}")
        updated: list[Any] = []
        for item in result:
            if isinstance(item, tuple):
                tmpl, lvl = item
                updated.append((pattern.sub(f"{{{slot}{suffix}}}", tmpl), lvl))
            else:
                updated.append(pattern.sub(f"{{{slot}{suffix}}}", item))
        result = updated
    return result


def main() -> None:
    pre_parser = __import__('argparse').ArgumentParser(add_help=False)
    pre_parser.add_argument('--story', default=None)
    pre_args, _ = pre_parser.parse_known_args()

    available = ability.list_ability_flavors(pre_args.story)
    if not available:
        print('No ability flavors found.')
        sys.exit(1)

    parser = build_parser(
        STORY, available, DEFAULT_LEVEL, _SLOT_EXAMPLE,
        epilog='The generic fallback is minimal. Use -f/--flavor to pick a theme, '
               'or -s/--slot with a technique group (status, melee, area, ...) '
               'to pool verbs across flavors.',
    )
    args = parser.parse_args()
    args.flavor_arg, args.final_count = parse_flavor_arg(
        parser, args, available, ability.categorized_ability_flavors(pre_args.story)
    )
    args.parsed_slot_sources, args.slot_modifiers, args.slot_queries, list_all = parse_slot_args(args.slot_sources)
    if list_all:
        print(f'Available slots: {", ".join(sorted(SLOTS))}')
        sys.exit(1)

    if args.required_slots is not None and '' in args.required_slots:
        print(f'Available slots (for -r): {", ".join(sorted(SLOTS))}')
        sys.exit(1)

    args.required_slots, require_sources = parse_require_args(args.required_slots)
    for slot, groups in require_sources.items():
        args.parsed_slot_sources.setdefault(slot, []).extend(groups)

    level_name, exact = parse_level(args.level)
    templates: list[tuple[str, str]] = filter_templates(
        load_templates_raw(ability._ability_dirs, pre_args.story, level_name, exact),
        args.required_slots,
    )

    if args.list_templates:
        list_templates(ability._ability_dirs, pre_args.story)
        sys.exit(0)

    resolved_template = resolve_template_idx(args.template, ability._ability_dirs, pre_args.story)
    args.template = resolved_template

    if args.enumerate_all:
        if args.template is None and not args.parsed_slot_sources and not args.all_force_destructive:
            print('--all without --template or --slot would produce enormous output and heavy resource use. '
                  'Add --all-force-destructive-i-know-what-im-doing if you really want this.')
            sys.exit(1)

    if args.flavor_arg == 'all':
        selected = list(available)
    elif args.flavor_arg:
        selected = [f.strip() for f in args.flavor_arg.split(',')]
    else:
        selected = list(available)

    # Auto-load flavors referenced by --slot
    for groups in args.parsed_slot_sources.values():
        for grp in groups:
            for src in grp:
                if src not in selected and src in available:
                    selected.append(src)

    if args.slot_queries:
        handle_slot_queries(
            args.slot_queries, selected, ability._ability_dirs,
            ALWAYS_LOAD, pre_args.story,
        )

    grammar = ability.load_ability_grammar(
        flavors=selected,
        slot_sources=args.parsed_slot_sources,
        story=pre_args.story,
        cull_sources=False,
    )

    templates = _apply_modifiers(templates, args.slot_modifiers)
    if args.template is not None:
        args.template = _apply_modifiers([args.template], args.slot_modifiers)[0]

    if args.inspect is not None:
        slot_filter = None if args.inspect == '__all__' else args.inspect
        inspect_grammar(grammar, selected, templates, SLOTS, all_entries=args.enumerate_all, slot_filter=slot_filter)
        sys.exit(0)

    label = ', '.join(selected)
    if args.parsed_slot_sources or args.slot_modifiers:
        parts = [f'{slot}={",".join("+".join(grp) for grp in groups)}' for slot, groups in args.parsed_slot_sources.items()]
        for slot, mods in args.slot_modifiers.items():
            parts.append(f'{slot}:{":".join(mods)}')
        label += f' (slots: {", ".join(parts)})'
    if args.level:
        label += f' (level={args.level})'

    if args.enumerate_all:
        tmpl_list: list[tuple[str, str | None]] = [args.template] if args.template else templates
        print(f'Flavor: {label}\n')
        if args.verbose:
            for tmpl, lvl in tmpl_list:
                for result_str, trace in expand_all_traced(tmpl, grammar):
                    print(f'  {result_str}')
                    print_verbose(tmpl, trace, level=lvl)
                    print()
            return
        outputs: set[str] = set()
        for tmpl, _lvl in tmpl_list:
            for result_str in expand_all(tmpl, grammar):
                outputs.add(result_str)
        for out in sorted(outputs):
            print(f'  {out}')
        print(f'\nTotal: {len(outputs)}')
        return

    print(f'Flavor: {label}\n')
    for i in range(args.final_count):
        tmpl, lvl = args.template if args.template else random.choice(templates)
        use_grammar = cull_grammar(grammar)
        if args.verbose:
            result, trace = expand_traced(tmpl, use_grammar)
        else:
            result = expand(tmpl, use_grammar)
        print(f'  {result}')
        if args.verbose:
            print_verbose(tmpl, trace, level=lvl)
            if i < args.final_count - 1:
                print()


if __name__ == '__main__':
    main()
