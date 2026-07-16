"""Fortune tool schemas and handlers for the orchestrator.

The director LLM exposes eight ``fortune_*`` tools (plus legacy aliases) for
objective randomness and grammar-based title/ability/name generation.  This
module owns their JSON schemas and argument handlers so
:meth:`Orchestrator.decide_next_turn` stays focused on decision flow.
"""

from __future__ import annotations

import json
import random
from typing import Any

from ara.config import AraSettings
from ara.llm.tools import ToolRegistry, tool
from ara.world import fortune


class FortuneTools:
    """Builds fortune tool schemas and registers their handlers.

    :param story: Story name used for per-story fortune data and flavors.
    :param config: Optional settings override for fortune data lookup.
    """

    def __init__(
        self, story: str | None = None, config: AraSettings | None = None
    ) -> None:
        """Create the fortune tool provider."""
        self.story = story
        self.config = config

    # ------------------------------------------------------------------ #
    # Schemas
    # ------------------------------------------------------------------ #

    @staticmethod
    def _flavor_list_blurb(categorized: dict[str, list[str]]) -> str:
        """Build a human-readable categorized flavor listing."""
        parts = []
        for cat in sorted(categorized):
            parts.append(f"{cat}: {', '.join(categorized[cat])}")
        return '; '.join(parts)

    def tools(self) -> list[dict[str, Any]]:
        """Return the fortune tool schemas in orchestrator prompt order."""
        fortune_roll_tool = tool(
            name='fortune_roll',
            description=(
                'Roll n dice of m faces. Use this to resolve uncertain actions, '
                'set success thresholds, or introduce objective randomness into the scene. '
                'Provide threshold to have the result evaluated automatically: '
                'total >= threshold -> pass_result; total < threshold -> fail_result.'
            ),
            properties={
                'n': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 100,
                    'description': 'Number of dice to roll.',
                },
                'm': {
                    'type': 'integer',
                    'minimum': 2,
                    'maximum': 1000,
                    'description': 'Number of faces per die.',
                },
                'threshold': {
                    'type': 'integer',
                    'description': 'Optional. Threshold to compare the total against. Result is auto-evaluated: total >= threshold means pass, total < threshold means fail.',
                },
                'pass_result': {
                    'type': 'string',
                    'description': 'Optional. What happens when total >= threshold, e.g. "survives" or "deals 3d10 damage".',
                },
                'fail_result': {
                    'type': 'string',
                    'description': 'Optional. What happens when total < threshold, e.g. "dies" or "misses".',
                },
            },
            required=['n', 'm'],
            strict=True,
        )

        _supported_distribs = sorted(fortune.supported_distributions())
        fortune_random_tool = tool(
            name='fortune_random',
            description=(
                'Sample a random value from a statistical distribution. '
                'Use this for weighted randomness, pacing, or probabilistic decisions.'
            ),
            properties={
                'distrib': {
                    'type': 'string',
                    'enum': _supported_distribs,
                    'description': f'Statistical distribution. Supported: {", ".join(_supported_distribs)}.',
                },
                'params': {
                    'type': 'object',
                    'description': (
                        'Distribution-specific parameters, e.g. {"mean": 0.5, "std": 0.15} for normal, '
                        '{"rate": 1.0} for exponential, {"alpha": 2, "beta": 5} for beta.'
                    ),
                },
            },
            required=['distrib'],
            strict=False,
        )

        fortune_iching_tool = tool(
            name='fortune_iching',
            description=(
                'Cast one of the 64 I-Ching hexagrams. Use this for omens, mood, '
                'or when the scene needs a symbolic direction that leaves room for interpretation. '
                'Pass verbose=true to also receive randomly-selected moving/changing lines.'
            ),
            properties={
                'verbose': {
                    'type': 'boolean',
                    'description': 'If true, include moving/changing lines in the result.',
                },
            },
            required=[],
            strict=False,
        )

        fortune_inspiration_tool = tool(
            name='fortune_inspiration',
            description=(
                'Receive a random word or short phrase. Use this as a creative seed '
                'to flavor the current scene.'
            ),
            properties={},
            required=[],
            strict=False,
        )

        fortune_suite_tool = tool(
            name='fortune_suite',
            description=(
                'Run several independent randomness tools at once: a die roll, '
                'a distribution sample, an I-Ching hexagram, and a random inspiration. '
                'Use this when you want multiple random inputs to consider together.'
            ),
            properties={},
            required=[],
            strict=False,
        )

        _story = self.story or ''
        _abl_cats = fortune.categorized_ability_flavors(_story)
        _t_cats = fortune.categorized_title_flavors(_story)
        _abl_blurb = self._flavor_list_blurb(_abl_cats)
        _t_blurb = self._flavor_list_blurb(_t_cats)
        _abl_slot_names = ", ".join(sorted(fortune.ABILITY_SLOTS))
        _t_slot_names = ", ".join(sorted(fortune.TITLE_SLOTS))

        fortune_title_tool = tool(
            name='fortune_title',
            description=(
                'Generate a random title, epithet, or honorific from the title grammar. '
                'Use this when a scene needs a fancy name for an NPC, item, faction, location, or concept. '
                f'Available flavors by category: {_t_blurb}.'
            ),
            properties={
                'flavor': {
                    'type': 'string',
                    'description': (
                        'Comma-separated flavor names to use, e.g. "fantasy,jrpg". '
                        'Omit to mix all available flavors.'
                    ),
                },
                'level': {
                    'type': 'string',
                    'description': (
                        'Complexity level: simple, moderate, complex, insane, or 0-3. '
                        'Append "!" for that level only. Omit for any complexity.'
                    ),
                },
                'template': {
                    'type': 'string',
                    'description': (
                        'Specific template such as "{adj} {noun} of {place}". '
                        f'Available slots: {_t_slot_names}. '
                        'Omit to pick a random template.'
                    ),
                },
                'require': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Only use templates that contain ALL of these slot names. '
                        f'Available slots: {_t_slot_names}. '
                        'Example: ["place", "suffix"] ensures the title includes a place and suffix. '
                        'Useful for constraining output shape without specifying a full template.'
                    ),
                },
                'slot': {
                    'type': 'object',
                    'description': (
                        'Per-slot source restrictions. Each key is a slot name, each value is a '
                        'list of source names. A source can be a flavor name, an internal group with ":" '
                        '(e.g. "nato:" to pull the merged nato group from all flavours), '
                        'or a literal value with "!" (e.g. "NATO!"). '
                        'Example: {"noun": ["foss"], "place": ["nato:", "NATO!"]}. '
                        'Omit to use all default sources.'
                    ),
                },
                'count': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 20,
                    'description': 'Number of titles to generate. Use this to generate all fighter titles in one call. Default 1.',
                },
                'verbose': {
                    'type': 'boolean',
                    'description': 'Include per-slot provenance (which flavor each part came from). Default false.',
                },
            },
            required=[],
            strict=False,
        )

        fortune_name_tool = tool(
            name='fortune_name',
            description=(
                'Generate a random human name by combining given and surname parts. '
                'Use this to name NPCs, background characters, or when a scene calls '
                'for a realistic human name on the fly.'
            ),
            properties={
                'style': {
                    'type': 'string',
                    'enum': ['random', 'simple', 'middle', 'spanish'],
                    'description': (
                        'Name style: "simple" (first+last, 2 parts), '
                        '"middle" (first+middle+last, 3 parts), '
                        '"spanish" (4-6 parts, long-form), '
                        '"random" (weighted distribution, default).'
                    ),
                },
                'n_parts': {
                    'type': 'integer',
                    'description': (
                        'Exact number of name parts. Overrides style if provided.'
                    ),
                },
            },
            required=[],
            strict=False,
        )

        fortune_ability_tool = tool(
            name='fortune_ability',
            description=(
                'Generate a random combat ability, spell, or technique from the ability grammar. '
                'Use this to assign thematic attacks to fighters, generate item effects, or '
                'name special moves. '
                'Each flavor contributes words to generic slots (domain, technique, verb, noun, '
                'prefix, suffix, adj, adj_sup). Delivery sources (melee, ranged, area, status) are '
                'cross-flavor aggregators: picking "melee" for the technique slot pulls fire-melee, '
                'ice-melee, corporate-melee etc. from every loaded flavor simultaneously. '
                f'Available flavors by category: {_abl_blurb}.'
            ),
            properties={
                'flavor': {
                    'type': 'string',
                    'description': (
                        'Comma-separated flavor names to load, e.g. "fire,melee". '
                        f'Available flavors by category: {_abl_blurb}. '
                        'Omit to load everything.'
                    ),
                },
                'level': {
                    'type': 'string',
                    'description': (
                        'Complexity level: simple (compound suffixes like Pyrothermia), '
                        'moderate (domain-technique like Inferno Slash), '
                        'complex (verb-of-noun like Audit of Flame), '
                        'insane (absurd combos). Append "!" for that level only. '
                        'Default mixes all levels up to complex.'
                    ),
                },
                'template': {
                    'type': 'string',
                    'description': (
                        'Specific template string such as "{domain} {technique}" or '
                        '"{verb} of {noun}" or "{adj} {noun} {roman_numeral}". '
                        f'Available slots: {_abl_slot_names}, ordinal, number, roman_numeral. '
                        'Omit to pick a random template.'
                    ),
                },
                'require': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Only use templates that contain ALL of these slot names. '
                        f'Available slots: {_abl_slot_names}, ordinal, number, roman_numeral. '
                        'Example: ["domain", "technique"] ensures the ability includes both. '
                        'Use "," for AND (all required), "+" for OR (any sufficient).'
                    ),
                },
                'slot': {
                    'type': 'object',
                    'description': (
                        'Per-slot source restrictions. Each key is a slot name, each value is a '
                        'list of source names. Sources can be flavour names or delivery groups. '
                        'Delivery groups (melee, ranged, area, status) aggregate internal groups '
                        'from every loaded flavour — "melee" for technique gives you fire:melee + '
                        'ice:melee + corporate:melee + … all at once. '
                        'Examples: {"technique": ["fire"]} restricts to fire-elemental techniques only. '
                        '{"domain": ["fire"], "technique": ["melee"]} creates fire-melee combos '
                        'like "Inferno Cleave" (fire domain + melee technique). '
                        '{"technique": ["area"]} gives area-of-effect style across all themes. '
                        f'Available slot names: {_abl_slot_names}. '
                        'Omit to use all default sources.'
                    ),
                },
                'count': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 20,
                    'description': 'Number of abilities to generate. Default 1.',
                },
                'verbose': {
                    'type': 'boolean',
                    'description': (
                        'Include per-slot provenance tracing showing which flavour contributed each word. '
                        'Default true.'
                    ),
                },
            },
            required=[],
            strict=False,
        )

        return [
            fortune_roll_tool,
            fortune_random_tool,
            fortune_iching_tool,
            fortune_inspiration_tool,
            fortune_suite_tool,
            fortune_title_tool,
            fortune_name_tool,
            fortune_ability_tool,
        ]

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def register(self, registry: ToolRegistry) -> None:
        """Register all fortune tool handlers and their legacy aliases."""
        registry.register('fortune_roll', self._fortune_roll_handler)
        registry.register('fortune_random', self._fortune_random_handler)
        registry.register('fortune_iching', self._fortune_iching_handler)
        registry.register(
            'fortune_inspiration', self._fortune_inspiration_handler
        )
        registry.register('fortune_suite', self._fortune_suite_handler)
        registry.register('fortune_title', self._fortune_title_handler)
        registry.register('fortune_name', self._fortune_name_handler)
        registry.register('fortune_ability', self._fortune_ability_handler)
        # Backward-compatible aliases for old tool names.
        registry.register('roll', self._fortune_roll_handler)
        registry.register('random', self._fortune_random_handler)
        registry.register('title', self._fortune_title_handler)
        registry.register('name', self._fortune_name_handler)

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #

    def _fortune_roll_handler(self, args: str) -> str:
        data = json.loads(args)
        n = int(data.get('n', 1))
        m = int(data.get('m', 100))
        if not 1 <= n <= 100:
            return f'Error: n must be between 1 and 100, got {n}.'
        if not 2 <= m <= 1000:
            return f'Error: m must be between 2 and 1000, got {m}.'
        rolls = [random.randint(1, m) for _ in range(n)]
        total = sum(rolls)
        result = f'Rolled {n}d{m}: {rolls} (sum: {total})'
        threshold = data.get('threshold')
        if threshold is not None:
            threshold = int(threshold)
            passed = total >= threshold
            result_text = data.get('pass_result', 'PASS')
            fail_text = data.get('fail_result', 'FAIL')
            outcome = result_text if passed else fail_text
            result += f' | threshold={threshold} -> {outcome} ({"pass" if passed else "fail"})'
        return result

    def _fortune_random_handler(self, args: str) -> str:
        data = json.loads(args)
        distrib = data.get('distrib', 'uniform')
        params = data.get('params') or {}
        try:
            value = fortune.sample_distribution(distrib, params)
        except ValueError as exc:
            return f'Error: {exc}'
        return f'Random {distrib} value: {value}'

    def _fortune_iching_handler(self, args: str) -> str:
        data = json.loads(args) if args.strip() else {}
        verbose = bool(data.get('verbose', False))
        h = fortune.cast_iching(self.story, config=self.config, verbose=verbose)
        chinese = h.get("chinese", "")
        judgment = h.get("judgment", "")
        lines = [
            f'I-Ching: {chinese}',
            f'Judgment: {judgment[:200]}...',
        ]
        if verbose:
            moving_lines = h.get("moving_lines", [])
            moving_desc = ", ".join(moving_lines) if moving_lines else "None"
            lines.append(f'Moving Lines: {moving_desc}')
        return "\n".join(lines)

    def _fortune_inspiration_handler(self, args: str) -> str:
        return f'Inspiration: {fortune.random_inspiration(self.story, config=self.config)}'

    def _fortune_suite_handler(self, args: str) -> str:
        suite = fortune.fortune_suite(self.story, config=self.config)
        iching = suite["iching"]
        iching_cn = iching.get("chinese", "")
        iching_judgment = iching.get("judgment", "")
        return (
            f'{suite["roll"]}\n'
            f'{suite["random"]}\n'
            f'I-Ching: {iching_cn} - {iching_judgment[:100]}...\n'
            f'Inspiration: {suite["inspiration"]}\n'
        )

    def _fortune_title_handler(self, args: str) -> str:
        data = json.loads(args) if args.strip() else {}
        flavor = data.get('flavor') or None
        level = data.get('level') or None
        template = data.get('template') or None
        required_slots = data.get('require') or None
        slot_sources_raw = data.get('slot') or data.get('slot_sources') or None
        count = int(data.get('count', 1))
        verbose = data.get('verbose', False)
        if isinstance(flavor, list):
            flavors = flavor
        elif flavor:
            flavors = [f.strip() for f in flavor.split(',') if f.strip()]
        else:
            flavors = None
        slot_sources = None
        if slot_sources_raw:
            if isinstance(slot_sources_raw, dict):
                slot_sources = {
                    k: v if isinstance(v, list) else [s.strip() for s in str(v).split(',') if s.strip()]
                    for k, v in slot_sources_raw.items()
                }
            elif isinstance(slot_sources_raw, str):
                try:
                    slot_sources = json.loads(slot_sources_raw)
                    if not isinstance(slot_sources, dict):
                        return f'Error: slot_sources must be a JSON object'
                    slot_sources = {
                        k: [s.strip() for s in v.split(',') if s.strip()]
                        if isinstance(v, str) else v
                        for k, v in slot_sources.items()
                    }
                except json.JSONDecodeError:
                    return f'Error: slot_sources must be valid JSON'
        if required_slots is not None and not isinstance(required_slots, list):
            return f'Error: require must be a list of slot names'
        try:
            import re as _re
            primary, fallback = fortune.title_dirs(str(self.story or ''))
            level_name, exact = fortune.resolve_level(level or "2")
            templates = fortune.load_title_templates(primary, fallback, level_name, exact)
            if required_slots:
                req = set(required_slots)
                templates = [t for t in templates if req.issubset(set(_re.findall(r'\{([^}+]+)\}', t)))]
                if not templates:
                    return f'Error: no templates contain all required slots: {", ".join(sorted(req))}'
            grammar = fortune.load_title_grammar(
                story=str(self.story or ''),
                flavors=flavors,
                slot_sources=slot_sources,
                cull_sources=False,
            )
            lines = []
            for _ in range(count):
                use_grammar = fortune.cull_grammar(grammar)
                tmpl = template if template else random.choice(templates)
                if verbose:
                    raw, trace = fortune.expand_traced(tmpl, use_grammar)
                    text = fortune.title_case(raw)
                    lines.append(f'Title: {text}')
                    for t in trace:
                        val = t['value'].strip()
                        if val:
                            slot_d = f'{t.get("parent","") + " -> " if t.get("parent") else ""}{t["slot"]}'; lines.append(f'  - {slot_d} [{t["source"]}] -> {val}')
                else:
                    text = fortune.title_case(
                        fortune.expand(tmpl, use_grammar)
                    )
                    lines.append(f'Title: {text}')
            return '\n'.join(lines)
        except ValueError as exc:
            return f'Error: {exc}'

    def _fortune_ability_handler(self, args: str) -> str:
        data = json.loads(args) if args.strip() else {}
        flavor = data.get('flavor') or None
        level = data.get('level') or None
        template = data.get('template') or None
        required_slots = data.get('require') or None
        slot_sources_raw = data.get('slot') or data.get('slot_sources') or None
        count = int(data.get('count', 1))
        verbose = data.get('verbose', True)
        if isinstance(flavor, list):
            flavors = flavor
        elif flavor:
            flavors = [f.strip() for f in flavor.split(',') if f.strip()]
        else:
            flavors = None
        slot_sources = None
        if slot_sources_raw:
            if isinstance(slot_sources_raw, dict):
                slot_sources = {
                    k: v if isinstance(v, list) else [s.strip() for s in str(v).split(',') if s.strip()]
                    for k, v in slot_sources_raw.items()
                }
            elif isinstance(slot_sources_raw, str):
                try:
                    slot_sources = json.loads(slot_sources_raw)
                    if not isinstance(slot_sources, dict):
                        return f'Error: slot_sources must be a JSON object'
                    slot_sources = {
                        k: [s.strip() for s in v.split(',') if s.strip()]
                        if isinstance(v, str) else v
                        for k, v in slot_sources.items()
                    }
                except json.JSONDecodeError:
                    return f'Error: slot_sources must be valid JSON'
        if required_slots is not None and not isinstance(required_slots, list):
            return f'Error: require must be a list of slot names'
        try:
            primary, fallback = fortune.ability_dirs(str(self.story or ''))
            level_name, exact = fortune.resolve_level(level or "2")
            templates = fortune.load_ability_templates(primary, fallback, level_name, exact)
            if required_slots:
                import re as _re2
                req = set(required_slots)
                templates = [t for t in templates if req.issubset(set(_re2.findall(r'\{([^}+]+)\}', t)))]
                if not templates:
                    return f'Error: no templates contain all required slots: {", ".join(sorted(req))}'
            grammar = fortune.load_ability_grammar(
                story=str(self.story or ''),
                flavors=flavors,
                slot_sources=slot_sources,
                cull_sources=False,
            )
            lines = []
            for _ in range(count):
                use_grammar = fortune.cull_grammar(grammar)
                tmpl = template if template else random.choice(templates)
                if verbose:
                    raw, trace = fortune.expand_traced(
                        tmpl, use_grammar,
                    )
                    text = fortune.title_case(raw)
                    lines.append(f'Ability: {text}')
                    for t in trace:
                        val = t['value'].strip()
                        if val:
                            slot_d = f'{t.get("parent","") + " -> " if t.get("parent") else ""}{t["slot"]}'; lines.append(f'  - {slot_d} [{t["source"]}] -> {val}')
                else:
                    text = fortune.title_case(
                        fortune.expand(
                            tmpl, use_grammar,
                        )
                    )
                    lines.append(f'Ability: {text}')
            return '\n'.join(lines)
        except ValueError as exc:
            return f'Error: {exc}'

    def _fortune_name_handler(self, args: str) -> str:
        data = json.loads(args) if args.strip() else {}
        style = data.get('style', 'random')
        n_parts = data.get('n_parts')
        if n_parts is not None:
            n_parts = int(n_parts)
        try:
            name = fortune.generate_name(style=style, n_parts=n_parts)
        except ValueError as exc:
            return f'Error: {exc}'
        return f'Name: {name}'
