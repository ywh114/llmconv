#!/usr/bin/env python3
"""CLI for sampling titles from the Ara title grammar."""

from __future__ import annotations

import random
import re
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

from ara.world import title

STORY = "title"
DEFAULT_LEVEL = "complex"
SLOTS = title._GENERIC_SLOTS
_SLOT_EXAMPLE = "--slot noun foss --slot prefix buzzword --slot place nato:"


def _resolve_sources(flavor_arg: str, all_flavors: list[str]) -> list[str]:
    """Resolve selected flavors from --flavor or positional arg."""
    if flavor_arg == 'all':
        return list(all_flavors)
    elif flavor_arg:
        return [f.strip() for f in flavor_arg.split(',')]
    return list(all_flavors)


def main() -> None:
    pre_parser = __import__('argparse').ArgumentParser(add_help=False)
    pre_parser.add_argument('--story', default=None)
    pre_args, _ = pre_parser.parse_known_args()

    available = title.list_title_flavors(pre_args.story)
    if not available:
        print('No title flavors found.')
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
    primary, fallback = title._title_dirs(pre_args.story)
    templates = filter_templates(
        load_templates_raw(title._title_dirs, pre_args.story, level_name, exact),
        args.required_slots,
    )

    if args.list_templates:
        list_templates(title._title_dirs, pre_args.story)
        sys.exit(0)

    args.template = resolve_template_idx(args.template, title._title_dirs, pre_args.story)

    selected = _resolve_sources(args.flavor_arg, available)

    # Auto-load flavors referenced by --slot
    for groups in args.parsed_slot_sources.values():
        for grp in groups:
            for src in grp:
                if src not in selected and src in available:
                    selected.append(src)

    if args.slot_queries:
        handle_slot_queries(
            args.slot_queries, selected, title._title_dirs,
            frozenset(title._ALWAYS_LOAD), pre_args.story,
        )

    grammar = title.build_grammar(
        selected, args.parsed_slot_sources, primary, fallback,
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
                for result_str, trace in title.expand_all_traced(tmpl, grammar):
                    cased = title.title_case(result_str)
                    print(f'  {cased}')
                    print_verbose(tmpl, trace)
                    print()
            return
        outputs: set[str] = set()
        for tmpl in tmpl_list:
            for raw in title.expand_all(tmpl, grammar):
                outputs.add(title.title_case(raw))
        for out in sorted(outputs):
            print(f'  {out}')
        print(f'\nTotal: {len(outputs)}')
        return

    print(f'Flavor: {label}\n')
    for i in range(args.final_count):
        tmpl = args.template if args.template else random.choice(templates)
        if args.verbose:
            raw, trace = title.expand_traced(tmpl, grammar)
        else:
            raw = title.expand(tmpl, grammar)
        result = title.title_case(raw)
        print(f'  {result}')
        if args.verbose:
            print_verbose(tmpl, trace)
            if i < args.final_count - 1:
                print()


if __name__ == '__main__':
    main()
