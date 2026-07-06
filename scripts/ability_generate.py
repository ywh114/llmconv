#!/usr/bin/env python3
"""CLI for sampling abilities from the Ara ability grammar."""

from __future__ import annotations

import random
import sys

from _grammar_cli import (
    build_parser,
    filter_templates,
    handle_slot_queries,
    inspect_grammar,
    list_templates,
    load_templates_raw,
    parse_flavor_arg,
    parse_level,
    parse_slot_args,
    print_verbose,
    resolve_template_idx,
)

from ara.world import ability
from ara.world.title import title_case, expand_traced, expand, expand_all, expand_all_traced

STORY = "ability"
DEFAULT_LEVEL = "complex"
SLOTS = ability._SLOTS
ALWAYS_LOAD = frozenset(ability._ALWAYS_LOAD)
_SLOT_EXAMPLE = "--slot domain fire --slot technique melee"


def main() -> None:
    pre_parser = __import__('argparse').ArgumentParser(add_help=False)
    pre_parser.add_argument('--story', default=None)
    pre_args, _ = pre_parser.parse_known_args()

    available = ability.list_ability_flavors(pre_args.story)
    if not available:
        print('No ability flavors found.')
        sys.exit(1)

    parser = build_parser(STORY, available, DEFAULT_LEVEL, _SLOT_EXAMPLE)
    args = parser.parse_args()
    args.flavor_arg, args.final_count = parse_flavor_arg(parser, args, available)
    args.parsed_slot_sources, args.slot_queries, list_all = parse_slot_args(args.slot_sources)
    if list_all:
        print(f'Available slots: {", ".join(sorted(SLOTS))}')
        sys.exit(1)

    if args.required_slots is not None and '' in args.required_slots:
        print(f'Available slots (for -r): {", ".join(sorted(SLOTS))}')
        sys.exit(1)

    level_name, exact = parse_level(args.level)
    templates = filter_templates(
        load_templates_raw(ability._ability_dirs, pre_args.story, level_name, exact),
        args.required_slots,
    )

    if args.list_templates:
        list_templates(ability._ability_dirs, pre_args.story)
        sys.exit(0)

    args.template = resolve_template_idx(args.template, ability._ability_dirs, pre_args.story)

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
    )

    if args.inspect is not None:
        slot_filter = None if args.inspect == '__all__' else args.inspect
        inspect_grammar(grammar, selected, templates, SLOTS, all_entries=args.enumerate_all, slot_filter=slot_filter)
        sys.exit(0)

    label = ', '.join(selected)
    if args.parsed_slot_sources:
        parts = [f'{slot}={",".join("+".join(grp) for grp in groups)}' for slot, groups in args.parsed_slot_sources.items()]
        label += f' (slots: {", ".join(parts)})'
    if args.level:
        label += f' (level={args.level})'

    if args.enumerate_all:
        restricting = args.template is not None or args.parsed_slot_sources or args.level is not None
        if not restricting and not args.all_force:
            print('--all without --template, --slot, or --level would produce enormous output. '
                  'Add --all-force if you really want this.')
            sys.exit(1)
        tmpl_list = [args.template] if args.template else templates
        print(f'Flavor: {label}\n')
        if args.verbose:
            for tmpl in tmpl_list:
                for result_str, trace in expand_all_traced(tmpl, grammar):
                    cased = title_case(result_str)
                    print(f'  {cased}')
                    print_verbose(tmpl, trace)
                    print()
            return
        outputs: set[str] = set()
        for tmpl in tmpl_list:
            for raw in expand_all(tmpl, grammar):
                outputs.add(title_case(raw))
        for out in sorted(outputs):
            print(f'  {out}')
        print(f'\nTotal: {len(outputs)}')
        return

    print(f'Flavor: {label}\n')
    for i in range(args.final_count):
        tmpl = args.template if args.template else random.choice(templates)
        if args.verbose:
            raw, trace = expand_traced(tmpl, grammar)
        else:
            raw = expand(tmpl, grammar)
        result = title_case(raw)
        print(f'  {result}')
        if args.verbose:
            print_verbose(tmpl, trace)
            if i < args.final_count - 1:
                print()


if __name__ == '__main__':
    main()
