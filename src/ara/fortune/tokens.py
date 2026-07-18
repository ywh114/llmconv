"""Token model and pipeline for fortune title/ability generation."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Token types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Literal:
    text: str


@dataclass(frozen=True)
class Slot:
    name: str
    modifiers: tuple[str, ...] = ()
    group: str | None = None


@dataclass(frozen=True)
class Word:
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Article:
    text: str


@dataclass(frozen=True)
class Affix:
    text: str
    side: str  # "left" for suffixes, "right" for prefixes
    meta: dict[str, Any] = field(default_factory=dict)


Token = Literal | Slot | Word | Article | Affix


# --------------------------------------------------------------------------- #
# Template tokenization
# --------------------------------------------------------------------------- #

_SLOT_RE = re.compile(r'\{([^}]+)\}')

# Obscure placeholder used by supersticky tokens to stick to the next word.
_SUPERSTICKY_PLACEHOLDER = '\uE000'


def tokenize(pattern: str) -> list[Token]:
    """Convert a template/pattern string into a list of tokens.

    Slot placeholders like ``{noun:plural}`` become ``Slot("noun", ("plural",))``.
    Slot names may contain ``+`` (e.g. ``{prefix+}``).  A trailing ``;GROUP``
    marks the slot as part of a selection group, e.g. ``{noun;0}``.
    """
    tokens: list[Token] = []
    pos = 0
    for m in _SLOT_RE.finditer(pattern):
        if m.start() > pos:
            tokens.append(Literal(pattern[pos : m.start()]))
        body = m.group(1)
        body, sep, group = body.rpartition(';')
        if not sep:
            body = m.group(1)
            group = ''
        group = group.strip() or None
        parts = body.split(':')
        name = parts[0]
        modifiers = tuple(p.strip() for p in parts[1:] if p.strip())
        tokens.append(Slot(name, modifiers, group))
        pos = m.end()
    if pos < len(pattern):
        tokens.append(Literal(pattern[pos:]))
    return tokens


# --------------------------------------------------------------------------- #
# Modifiers and entry selection
# --------------------------------------------------------------------------- #

def _entry_text(entry: Any) -> str | None:
    if isinstance(entry, dict):
        if 'value' in entry:
            return entry['value']
        if 'patterns' in entry:
            return None
    return str(entry)


def _entry_matches_affinity(
    entry: Any,
    requires_prefixable: bool,
    requires_suffixible: bool,
) -> bool:
    """Return True if an entry can accept the required adjacent affixes."""
    if not requires_prefixable and not requires_suffixible:
        return True
    meta = entry if isinstance(entry, dict) else {}
    if requires_prefixable and not meta.get('prefixable', True):
        return False
    if requires_suffixible and not meta.get('suffixible', True):
        return False
    return True


def select_entry(
    entries: list[Any],
    modifiers: tuple[str, ...],
    slot: str,
    requires_prefixable: bool = False,
    requires_suffixible: bool = False,
) -> Any:
    """Pick an entry, honoring numeric constraints and affix affinity."""
    if not entries:
        raise ValueError(f'No entries for slot {slot}')
    candidates = list(entries)
    if requires_prefixable or requires_suffixible:
        compatible = [
            e
            for e in candidates
            if _entry_matches_affinity(
                e, requires_prefixable, requires_suffixible
            )
        ]
        if compatible:
            candidates = compatible
    return random.choice(candidates)


# --------------------------------------------------------------------------- #
# Default morphology
# --------------------------------------------------------------------------- #

_VOWELS = set('aeiou')


def default_plural(word: str) -> str:
    if not word:
        return word
    w = word.lower()
    # Irregular defaults that are too common to require metadata
    if w in {'goose', 'mouse', 'louse'}:
        return (
            word[:-3] + 'eese' if word.endswith('oose') else word[:-4] + 'ice'
        )
    if w.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'
    if w.endswith('y') and len(w) > 1 and w[-2] not in _VOWELS:
        return word[:-1] + 'ies'
    if w.endswith('f') and len(w) > 1:
        return word[:-1] + 'ves'
    if w.endswith('fe'):
        return word[:-2] + 'ves'
    return word + 's'


def default_gerund(word: str) -> str:
    if not word:
        return word
    w = word.lower()
    # silent e: make -> making, but not ee
    if w.endswith('e') and not w.endswith('ee'):
        return word[:-1] + 'ing'
    # consonant-vowel-consonant doubling
    if (
        len(w) >= 3
        and w[-1] not in _VOWELS | {'w', 'x', 'y'}
        and w[-2] in _VOWELS
        and w[-3] not in _VOWELS
    ):
        return word + word[-1] + 'ing'
    return word + 'ing'


def default_past(word: str) -> str:
    if not word:
        return word
    w = word.lower()
    if w.endswith('e'):
        return word + 'd'
    if w.endswith('y') and len(w) > 1 and w[-2] not in _VOWELS:
        return word[:-1] + 'ied'
    if (
        len(w) >= 3
        and w[-1] not in _VOWELS | {'w', 'x', 'y'}
        and w[-2] in _VOWELS
        and w[-3] not in _VOWELS
    ):
        return word + word[-1] + 'ed'
    return word + 'ed'


def default_superlative(word: str) -> str:
    if not word:
        return word
    w = word.lower()
    if w.endswith('e'):
        return word + 'st'
    if w.endswith('y') and len(w) > 1 and w[-2] not in _VOWELS:
        return word[:-1] + 'iest'
    if (
        len(w) >= 3
        and w[-1] not in _VOWELS | {'w', 'x', 'y'}
        and w[-2] in _VOWELS
        and w[-3] not in _VOWELS
    ):
        return word + word[-1] + 'est'
    return word + 'est'


def default_comparative(word: str) -> str:
    if not word:
        return word
    w = word.lower()
    if w.endswith('e'):
        return word + 'r'
    if w.endswith('y') and len(w) > 1 and w[-2] not in _VOWELS:
        return word[:-1] + 'ier'
    if (
        len(w) >= 3
        and w[-1] not in _VOWELS | {'w', 'x', 'y'}
        and w[-2] in _VOWELS
        and w[-3] not in _VOWELS
    ):
        return word + word[-1] + 'er'
    return word + 'er'


def default_possessive(word: str) -> str:
    if not word:
        return word
    if word.endswith('s'):
        return word + "'"
    return word + "'s"


def _transform_value(value: str, modifier: str, meta: dict[str, Any]) -> str:
    pos = meta.get('pos', 'verb')
    if modifier == 'plural':
        return meta.get('plural') or default_plural(value)
    if modifier == 'gerund':
        if pos == 'noun':
            return value
        return meta.get('gerund') or default_gerund(value)
    if modifier == 'past':
        if pos == 'noun':
            return value
        return meta.get('past') or default_past(value)
    if modifier == 'noun':
        if pos == 'noun':
            return value
        return meta.get('noun_form') or value
    if modifier == 'sup':
        return meta.get('superlative') or default_superlative(value)
    if modifier == 'comp':
        return meta.get('comparative') or default_comparative(value)
    if modifier == 'possessive':
        return meta.get('possessive') or default_possessive(value)
    raise ValueError(f"Unknown modifier '{modifier}' for value '{value}'")


# --------------------------------------------------------------------------- #
# Entry → tokens
# --------------------------------------------------------------------------- #

_ARTICLE_RE = re.compile(r'^(the|a|an)\s+', re.IGNORECASE)


def _split_article(text: str) -> tuple[str | None, str]:
    m = _ARTICLE_RE.match(text)
    if m:
        return m.group(1).lower(), text[m.end() :]
    return None, text


def _value_to_tokens(value: str, meta: dict[str, Any]) -> list[Token]:
    """Tokenize a value string, attaching metadata to Word tokens.

    Plain text is split into words and spacing literals so title-casing works
    per word.  Values may also contain slot placeholders, e.g.
    ``"A({number})"``.
    """
    tokens: list[Token] = []
    for t in tokenize(value):
        if isinstance(t, Literal):
            # Preserve spacing so render_tokens can rebuild the original gaps.
            for part in re.split(r'(\s+)', t.text):
                if not part:
                    continue
                if part.isspace():
                    tokens.append(Literal(part))
                else:
                    tokens.append(Word(part, meta.copy()))
        elif isinstance(t, Word):
            tokens.append(Word(t.text, meta.copy()))
        else:
            tokens.append(t)
    return tokens


def entry_to_tokens(
    entry: Any,
    slot: str,
    modifiers: tuple[str, ...],
) -> tuple[list[Token], dict[str, Any] | None]:
    """Convert a single grammar entry into a token sequence.

    Returns ``(tokens, trace_entry)`` where ``trace_entry`` is ``None`` for
    pattern entries (their recursive expansion supplies its own trace).

    ``prefix`` entries become right-attaching affixes; ``suffix`` entries become
    left-attaching affixes.  All other slots produce ordinary words.
    """
    if isinstance(entry, dict) and 'patterns' in entry:
        # Pattern entries are expanded recursively by the caller.
        return [Slot(slot, modifiers)], None

    is_affix = False
    side = ''
    if slot == 'prefix':
        is_affix = True
        side = 'right'
    elif slot == 'suffix':
        is_affix = True
        side = 'left'

    if isinstance(entry, dict) and 'value' in entry:
        meta = {
            k: v
            for k, v in entry.items()
            if k not in ('value', 'patterns', '_source')
        }
        meta.setdefault('sticky', True)
        value = entry['value']
        article = entry.get('article')
        if not article:
            article, value = _split_article(value)

        for mod in modifiers:
            value = _transform_value(value, mod, meta)
            article = None

        tokens: list[Token] = []
        if is_affix:
            tokens.append(Affix(value, side, meta.copy()))
        else:
            if article:
                tokens.append(Article(article))
            tokens.extend(_value_to_tokens(value, meta))
        trace = {
            'slot': slot,
            'value': value,
            'source': entry.get('_source', '?'),
            'modifiers': modifiers,
        }
        return tokens, trace

    # Plain string / other entry.
    if is_affix:
        return [Affix(str(entry), side)], {
            'slot': slot,
            'value': str(entry),
            'source': '?',
            'modifiers': modifiers,
        }
    return [Word(str(entry), {'sticky': True})], {
        'slot': slot,
        'value': str(entry),
        'source': '?',
        'modifiers': modifiers,
    }


# --------------------------------------------------------------------------- #
# Expansion
# --------------------------------------------------------------------------- #


def _single_slot_name(pattern: str) -> str | None:
    """If a pattern is exactly one slot placeholder, return its name."""
    toks = tokenize(pattern)
    if len(toks) == 1 and isinstance(toks[0], Slot):
        return toks[0].name
    return None


def _is_prefix_source(token: Token) -> bool:
    """True if the token represents a prefix affix adjacent to a target word."""
    if isinstance(token, Slot):
        return token.name in {'prefix', 'prefix+'}
    if isinstance(token, Affix):
        return token.side == 'right'
    return False


def _is_suffix_source(token: Token) -> bool:
    """True if the token represents a suffix affix adjacent to a target word."""
    if isinstance(token, Slot):
        return token.name in {'suffix', '_suffix+'}
    if isinstance(token, Affix):
        return token.side == 'left'
    return False


def _is_value_entry(entry: Any) -> bool:
    """True for plain string or dict entries with a ``value`` key."""
    return not (isinstance(entry, dict) and 'patterns' in entry)


def _resolve_group_entry(
    entries: list[Any],
    modifiers: tuple[str, ...],
    slot: str,
    group: str,
    groups: dict[str, dict[str, Any]],
    group_used: dict[str, set[str]],
    requires_prefixable: bool,
    requires_suffixible: bool,
) -> Any:
    """Pick or reuse a grouped entry, honoring cross-slot disjointness.

    For primitive slots this selects a concrete value entry.  For composite
    (pattern-only) slots it falls back to reusing the chosen pattern entry,
    so later occurrences expand with the same pattern shape.
    """
    slot_groups = groups.setdefault(group, {})
    if slot in slot_groups:
        return slot_groups[slot]

    value_entries = [e for e in entries if _is_value_entry(e)]
    candidates = value_entries if value_entries else entries

    used = group_used.get(group)
    if used:
        unused = [e for e in candidates if _entry_text(e) not in used]
        if unused:
            candidates = unused

    entry = select_entry(
        candidates,
        modifiers,
        slot,
        requires_prefixable=requires_prefixable,
        requires_suffixible=requires_suffixible,
    )
    slot_groups[slot] = entry
    group_used.setdefault(group, set()).add(_entry_text(entry))
    return entry


def _synthetic_group(
    outer_group: str, composite_slot: str, inner_slot: str
) -> str:
    """Return a deterministic group id for an inner slot of a grouped composite."""
    return f'__auto__{outer_group}__{composite_slot}__{inner_slot}'


def _apply_composite_groups(
    sub_tokens: list[Token],
    outer_group: str,
    composite_slot: str,
) -> list[Token]:
    """Give ungrouped inner slots of a composite deterministic synthetic groups."""
    result: list[Token] = []
    for t in sub_tokens:
        if isinstance(t, Slot) and t.group is None:
            result.append(
                Slot(
                    t.name,
                    t.modifiers,
                    _synthetic_group(outer_group, composite_slot, t.name),
                )
            )
        else:
            result.append(t)
    return result


def _apply_last_slot_modifier(
    sub_tokens: list[Token],
    modifiers: tuple[str, ...],
) -> list[Token]:
    """Forward outer modifiers to the last inner slot of a composite pattern."""
    if not modifiers:
        return sub_tokens
    last_idx: int | None = None
    for i, t in enumerate(sub_tokens):
        if isinstance(t, Slot):
            last_idx = i
    if last_idx is None:
        return sub_tokens
    target = sub_tokens[last_idx]
    assert isinstance(target, Slot)
    new_modifiers = modifiers + target.modifiers
    sub_tokens[last_idx] = Slot(target.name, new_modifiers, target.group)
    return sub_tokens


def expand_tokens(
    tokens: list[Token],
    grammar: dict[str, list],
    depth: int = 0,
    max_depth: int = 10,
    _parent: str = '',
    _groups: dict[str, dict[str, Any]] | None = None,
    _group_used: dict[str, set[str]] | None = None,
    _group_patterns: dict[tuple[str, str], str] | None = None,
) -> tuple[list[Token], list[dict]]:
    """Recursively expand Slot tokens into token sequences.

    Returns ``(tokens, trace)``.  Pattern entries produce nested traces with
    ``parent`` set to the containing slot name.  Value entries that contain
    embedded slot placeholders are expanded in subsequent passes.
    """
    if depth >= max_depth:
        return tokens, []

    groups = _groups if _groups is not None else {}
    group_used = _group_used if _group_used is not None else {}
    group_patterns = _group_patterns if _group_patterns is not None else {}
    trace: list[dict] = []
    result: list[Token] = []

    for i, token in enumerate(tokens):
        if not isinstance(token, Slot):
            result.append(token)
            continue

        slot = token.name
        entries = grammar.get(slot, [])
        if not entries:
            result.append(token)
            continue

        requires_prefixable = i > 0 and _is_prefix_source(tokens[i - 1])
        requires_suffixible = i + 1 < len(tokens) and _is_suffix_source(
            tokens[i + 1]
        )
        if token.group:
            entry = _resolve_group_entry(
                entries,
                token.modifiers,
                slot,
                token.group,
                groups,
                group_used,
                requires_prefixable,
                requires_suffixible,
            )
        else:
            entry = select_entry(
                entries,
                token.modifiers,
                slot,
                requires_prefixable=requires_prefixable,
                requires_suffixible=requires_suffixible,
            )

        if isinstance(entry, dict) and 'patterns' in entry:
            group_key = (token.group, slot) if token.group else None
            if group_key is not None and group_key in group_patterns:
                pattern = group_patterns[group_key]
            else:
                pattern = random.choice(entry['patterns'])
                if group_key is not None:
                    group_patterns[group_key] = pattern

            sub_tokens = tokenize(pattern)
            if token.group:
                sub_tokens = _apply_composite_groups(
                    sub_tokens, token.group, slot
                )
            sub_tokens = _apply_last_slot_modifier(sub_tokens, token.modifiers)

            expanded, sub_trace = expand_tokens(
                sub_tokens,
                grammar,
                depth + 1,
                max_depth,
                _parent=slot,
                _groups=groups,
                _group_used=group_used,
                _group_patterns=group_patterns,
            )
            for tr in sub_trace:
                tr.setdefault('parent', slot)
            result.extend(expanded)
            trace.extend(sub_trace)
        else:
            sub_tokens, tr = entry_to_tokens(entry, slot, token.modifiers)
            if tr:
                if _parent:
                    tr['parent'] = _parent
                trace.append(tr)
            result.extend(sub_tokens)

    # Expand any slot placeholders that were embedded in value strings.
    if any(isinstance(t, Slot) for t in result):
        final, nested_trace = expand_tokens(
            result,
            grammar,
            depth + 1,
            max_depth,
            _parent,
            _groups=groups,
            _group_used=group_used,
            _group_patterns=group_patterns,
        )
        return final, trace + nested_trace

    return result, trace


def expand(
    pattern: str,
    grammar: dict[str, list],
    depth: int = 0,
    max_depth: int = 10,
) -> str:
    """Expand a pattern to a string (untraced)."""
    tokens = tokenize(pattern)
    expanded, _ = expand_tokens(tokens, grammar, depth, max_depth)
    processed = render_pipeline(expanded)
    return processed


def expand_traced(
    pattern: str,
    grammar: dict[str, list],
    depth: int = 0,
    max_depth: int = 10,
    _parent: str = '',
) -> tuple[str, list[dict]]:
    """Expand a pattern to a string with provenance trace."""
    tokens = tokenize(pattern)
    expanded, trace = expand_tokens(tokens, grammar, depth, max_depth, _parent)
    processed = render_pipeline(expanded)
    return processed, trace


def _expand_entry_all(
    entry: Any,
    slot: str,
    modifiers: tuple[str, ...],
    grammar: dict[str, list],
    depth: int,
    max_depth: int,
    _groups: dict[str, dict[str, Any]] | None = None,
    _group_used: dict[str, set[str]] | None = None,
) -> list[tuple[list[Token], list[dict]]]:
    """Return all expansions of a single grammar entry as (tokens, trace)."""
    if depth >= max_depth:
        return [([], [])]

    src = entry.get('_source', '?') if isinstance(entry, dict) else '?'

    if isinstance(entry, dict) and 'patterns' in entry:
        results: list[tuple[list[Token], list[dict]]] = []
        for sub_pattern in entry['patterns']:
            inner = _single_slot_name(sub_pattern)
            if inner is not None:
                sub_toks = [Slot(inner, modifiers)]
            else:
                sub_toks = tokenize(sub_pattern)
            for sub_tokens, sub_trace in expand_all_tokens(
                sub_toks,
                grammar,
                depth + 1,
                max_depth,
                _groups=_groups,
                _group_used=_group_used,
            ):
                for tr in sub_trace:
                    tr.setdefault('parent', slot)
                results.append((sub_tokens, sub_trace))
        return results

    tokens, tr = entry_to_tokens(entry, slot, modifiers)
    trace = [tr] if tr else []
    return [(tokens, trace)]


def _copy_groups(
    groups: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {gid: dict(mapping) for gid, mapping in groups.items()}


def _copy_group_used(group_used: dict[str, set[str]]) -> dict[str, set[str]]:
    return {gid: set(values) for gid, values in group_used.items()}


def _expand_all_tokens_impl(
    tokens: list[Token],
    grammar: dict[str, list],
    pos: int,
    depth: int,
    max_depth: int,
    groups: dict[str, dict[str, Any]],
    group_used: dict[str, set[str]],
) -> list[tuple[list[Token], list[dict]]]:
    """Enumerate every expansion of a token list, respecting selection groups."""
    if pos >= len(tokens):
        return [([], [])]

    token = tokens[pos]
    if not isinstance(token, Slot):
        rest = _expand_all_tokens_impl(
            tokens, grammar, pos + 1, depth, max_depth, groups, group_used
        )
        return [([token] + sub_toks, sub_trace) for sub_toks, sub_trace in rest]

    slot = token.name
    entries = grammar.get(slot, [])
    if not entries:
        rest = _expand_all_tokens_impl(
            tokens, grammar, pos + 1, depth, max_depth, groups, group_used
        )
        return [([token] + sub_toks, sub_trace) for sub_toks, sub_trace in rest]

    requires_prefixable = pos > 0 and _is_prefix_source(tokens[pos - 1])
    requires_suffixible = pos + 1 < len(tokens) and _is_suffix_source(
        tokens[pos + 1]
    )

    if token.group:
        group = token.group
        slot_groups = groups.setdefault(group, {})
        if slot in slot_groups:
            chosen_entries = [slot_groups[slot]]
        else:
            value_entries = [
                e for e in entries if _is_value_entry(e)
            ] or entries
            used = group_used.get(group, set())
            unused = [e for e in value_entries if _entry_text(e) not in used]
            chosen_entries = unused if unused else value_entries
    else:
        chosen_entries = list(entries)

    if requires_prefixable or requires_suffixible:
        compatible = [
            e
            for e in chosen_entries
            if _entry_matches_affinity(
                e, requires_prefixable, requires_suffixible
            )
        ]
        if compatible:
            chosen_entries = compatible

    results: list[tuple[list[Token], list[dict]]] = []
    for entry in chosen_entries:
        next_groups = groups
        next_group_used = group_used
        if token.group and slot not in groups[token.group]:
            next_groups = _copy_groups(groups)
            next_groups.setdefault(token.group, {})[slot] = entry
            next_group_used = _copy_group_used(group_used)
            next_group_used.setdefault(token.group, set()).add(
                _entry_text(entry)
            )

        for sub_tokens, sub_trace in _expand_entry_all(
            entry, slot, token.modifiers, grammar, depth, max_depth
        ):
            for rest_tokens, rest_trace in _expand_all_tokens_impl(
                tokens,
                grammar,
                pos + 1,
                depth,
                max_depth,
                next_groups,
                next_group_used,
            ):
                results.append(
                    (sub_tokens + rest_tokens, sub_trace + rest_trace)
                )
    return results


def expand_all_tokens(
    tokens: list[Token],
    grammar: dict[str, list],
    depth: int = 0,
    max_depth: int = 10,
    _groups: dict[str, dict[str, Any]] | None = None,
    _group_used: dict[str, set[str]] | None = None,
) -> list[tuple[list[Token], list[dict]]]:
    """Enumerate every expansion of a token list."""
    if depth >= max_depth:
        return [(tokens, [])]
    groups = _groups if _groups is not None else {}
    group_used = _group_used if _group_used is not None else {}
    return _expand_all_tokens_impl(
        tokens, grammar, 0, depth, max_depth, groups, group_used
    )


def expand_all(
    pattern: str,
    grammar: dict[str, list],
    depth: int = 0,
    max_depth: int = 10,
) -> list[str]:
    """Enumerate every expansion of a pattern."""
    tokens = tokenize(pattern)
    results: list[str] = []
    for expanded, _ in expand_all_tokens(tokens, grammar, depth, max_depth):
        results.append(render_pipeline(expanded))
    return results


def expand_all_traced(
    pattern: str,
    grammar: dict[str, list],
    depth: int = 0,
    max_depth: int = 10,
) -> list[tuple[str, list[dict]]]:
    """Enumerate every expansion of a pattern with provenance traces."""
    tokens = tokenize(pattern)
    results: list[tuple[str, list[dict]]] = []
    for expanded, trace in expand_all_tokens(tokens, grammar, depth, max_depth):
        results.append((render_pipeline(expanded), trace))
    return results


# --------------------------------------------------------------------------- #
# Rendering pipeline
# --------------------------------------------------------------------------- #

_SMALL_WORDS = {
    'a',
    'an',
    'as',
    'the',
    'and',
    'but',
    'or',
    'nor',
    'for',
    'yet',
    'so',
    'at',
    'around',
    'by',
    'after',
    'along',
    'from',
    'of',
    'on',
    'in',
    'to',
    'with',
    'without',
    'within',
    'upon',
    'over',
    'under',
    'against',
    'beyond',
    'through',
    'between',
    'among',
    'near',
    'before',
    'behind',
    'below',
    'beneath',
    'beside',
    'during',
    'inside',
    'outside',
    'into',
    'onto',
    'about',
    'above',
    'across',
    'after',
    'around',
    'despite',
    'except',
    'like',
    'off',
    'past',
    'since',
    'throughout',
    'till',
    'toward',
    'towards',
    'until',
    'up',
    'via',
    'versus',
    'vs',
    'per',
    'pro',
    'qua',
}

# Preserved uppercase tokens: SI units and a few math/group symbols.
_PRESERVE_CASE = {
    'm',
    'g',
    's',
    'A',
    'K',
    'mol',
    'cd',
    'N',
    'J',
    'W',
    'Pa',
    'Hz',
    'V',
    'C',
    'Ω',
    'Z',
    'S',
    'D',
    'A',
    'GL',
    'SL',
    'O',
    'SO',
    'ħ',
    'π',
    'I',
    'II',
    'III',
    'IV',
    'V',
    'VI',
    'VII',
    'VIII',
    'IX',
    'X',
    'XI',
    'XII',
    'L',
    'C',
    'D',
    'M',
}


def _is_non_sticky_word(token: Token) -> bool:
    return isinstance(token, Word) and not token.meta.get('sticky', True)


def _gap_needs_space(left: Token, right: Token) -> bool:
    """A gap needs a separator if either side is a non-sticky Word."""
    return _is_non_sticky_word(left) or _is_non_sticky_word(right)


def _is_preserved_separator(literals: list[Literal]) -> bool:
    """True when the literals already separate two content tokens."""
    if not literals:
        return False
    text = ''.join(l.text for l in literals)
    if text and text[-1].isspace():
        return True
    stripped = text.strip()
    if not stripped:
        return True
    # Any non-whitespace char that isn't a hyphen glue means it's a real separator.
    return any(ch not in '-\u2013\u2014' for ch in stripped)


def _is_hyphen_glue(literals: list[Literal]) -> bool:
    """True when the only non-whitespace content between tokens is hyphens."""
    if not literals:
        return False
    text = ''.join(l.text for l in literals)
    stripped = text.strip()
    return bool(stripped) and all(ch in '-\u2013\u2014' for ch in stripped)


def apply_sticky(tokens: list[Token]) -> list[Token]:
    """Force spaces around non-sticky Word tokens.

    ``sticky=True`` (default) means a token may stick to neighbors with no
    separator. ``sticky=False`` means a separator is required. A bare hyphen
    between a non-sticky token and its neighbor is treated as glue and is
    replaced by a space; commas, colons, whitespace, and other punctuation are
    left alone.
    """
    content_types = (Word, Affix, Article)
    indices = [i for i, t in enumerate(tokens) if isinstance(t, content_types)]
    result = list(tokens)
    # Process pairs in reverse so earlier replacements do not shift later indices.
    for left_idx, right_idx in zip(reversed(indices[:-1]), reversed(indices[1:])):
        left = result[left_idx]
        right = result[right_idx]
        between = result[left_idx + 1 : right_idx]
        if any(not isinstance(t, Literal) for t in between):
            continue
        # Supersticky tokens stick to the next word via an invisible placeholder.
        if getattr(left, 'meta', {}).get('supersticky'):
            if not between or ''.join(l.text for l in between).strip() != _SUPERSTICKY_PLACEHOLDER:
                result[left_idx + 1 : right_idx] = [Literal(_SUPERSTICKY_PLACEHOLDER)]
            continue
        if not _gap_needs_space(left, right):
            continue
        if _is_preserved_separator(between):
            continue
        if _is_hyphen_glue(between):
            result[left_idx + 1 : right_idx] = [Literal(' ')]
        elif not between:
            result.insert(right_idx, Literal(' '))
    return result


def attach_affixes(tokens: list[Token]) -> list[Token]:
    """Merge Affix tokens with adjacent Word tokens.

    Runs until stable so chains like ``{prefix}{prefix}{noun}`` collapse
    into a single word.  Honors ``prefixable``/``suffixible`` metadata on the
    target word; if the target rejects the affix, the affix is dropped.
    """
    changed = True
    while changed:
        changed = False
        result: list[Token] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if isinstance(token, Affix):
                if token.side == 'left':
                    if not result:
                        # A suffix with nothing to attach to is dropped.
                        i += 1
                        continue
                    if isinstance(result[-1], Word):
                        prev = result[-1]
                        if prev.meta.get('suffixible', True):
                            result[-1] = Word(prev.text + token.text, prev.meta)
                            changed = True
                        # If suffixible is false, drop the suffix.
                        i += 1
                        continue
                    # A left affix after another affix: merge leftward into a combined affix.
                    if isinstance(result[-1], Affix):
                        prev = result.pop()
                        merged_meta = {**prev.meta, **token.meta}
                        result.append(
                            Affix(prev.text + token.text, side='left', meta=merged_meta)
                        )
                        changed = True
                        i += 1
                        continue
                    # Suffix cannot attach to a non-word (e.g. across a space): drop it.
                    i += 1
                    continue
                if token.side == 'right' and i + 1 < len(tokens):
                    nxt = tokens[i + 1]
                    if isinstance(nxt, Word):
                        if nxt.meta.get('prefixable', True):
                            result.append(Word(token.text + nxt.text, nxt.meta))
                            changed = True
                            i += 2
                        else:
                            # If prefixable is false, drop the prefix.
                            i += 1
                        continue
                    # A right affix before another affix: merge rightward.
                    if isinstance(nxt, Affix):
                        merged_meta = {**token.meta, **nxt.meta}
                        result.append(
                            Affix(token.text + nxt.text, side='right', meta=merged_meta)
                        )
                        changed = True
                        i += 2
                        continue
                # No attachment possible: render as word.
                result.append(Word(token.text))
            else:
                result.append(token)
            i += 1
        tokens = result
    return tokens


# Articles are kept after these tokens.
_KEEP_ARTICLE_AFTER = frozenset(
    {
        'of',
        'from',
        'in',
        'on',
        'at',
        'to',
        'by',
        'for',
        'with',
        'without',
        'upon',
        'over',
        'under',
        'against',
        'beyond',
        'through',
        'between',
        'among',
        'within',
        'across',
        'behind',
        'beside',
        'near',
        'inside',
        'outside',
        'toward',
        'towards',
        'about',
        'into',
        'onto',
        'before',
        'after',
        'along',
        'around',
        'below',
        'beneath',
        'during',
        'throughout',
        'via',
        'past',
        'despite',
        'except',
        'like',
        'and',
        'or',
        'nor',
        'yet',
        # Verbs used by the parenthetical templates ("(hates {place})" etc.).
        'hates',
        'loves',
    }
)


def apply_articles(tokens: list[Token]) -> list[Token]:
    """Drop articles that do not follow a keep-article word/token."""
    result: list[Token] = []
    for i, token in enumerate(tokens):
        if isinstance(token, Article):
            if i == 0:
                result.append(token)
                continue
            prev = tokens[i - 1]
            prev_text = ''
            if isinstance(prev, Word):
                prev_text = prev.text
            elif isinstance(prev, Literal):
                # Compare the trailing word only; punctuation glued to it
                # (e.g. the "(" in " (hates ") must not break the lookup.
                m = re.search(r"[A-Za-z']+$", prev.text.rstrip())
                prev_text = m.group(0) if m else ''
            if prev_text.lower() in _KEEP_ARTICLE_AFTER:
                result.append(token)
            # Otherwise drop.
        else:
            result.append(token)
    return result


def title_case_tokens(tokens: list[Token]) -> list[Token]:
    """Apply title casing to Word and leading Article tokens."""
    out: list[Token] = []
    is_first_word = True
    for token in tokens:
        if isinstance(token, Word):
            text = token.text
            lowered = text.lower()
            prev = out[-1] if out else None
            is_compound_continuation = isinstance(prev, Word)
            if token.meta.get('lowercase'):
                pass
            elif text in _PRESERVE_CASE:
                pass
            elif is_compound_continuation and not token.meta.get('proper_noun'):
                text = lowered
            elif is_first_word or lowered not in _SMALL_WORDS:
                text = text[:1].upper() + text[1:]
            out.append(Word(text, token.meta))
            is_first_word = False
        elif isinstance(token, Article):
            text = token.text
            if is_first_word:
                text = text[:1].upper() + text[1:]
            # Otherwise leave articles lowercase (e.g. "of the camp").
            out.append(Article(text))
            is_first_word = False
        elif isinstance(token, Literal):
            out.append(token)
            # A literal ending with sentence punctuation can reset first-word state.
            stripped = token.text.rstrip()
            if stripped and stripped[-1] in ':;—–-':
                is_first_word = True
        else:
            out.append(token)
            if getattr(token, 'meta', {}).get('supersticky'):
                is_first_word = False
    return out


def render_tokens(tokens: list[Token]) -> str:
    """Join tokens into a string.

    Spaces come from ``Literal`` tokens in the original template/value.  The
    only automatic spacing is around ``Article`` tokens (e.g. "the campus")
    because articles are split out of value strings and have no literal gap.
    """
    parts: list[str] = []
    prev: Token | None = None
    for token in tokens:
        if isinstance(token, Literal):
            if token.text == _SUPERSTICKY_PLACEHOLDER:
                continue
            parts.append(token.text)
        elif isinstance(token, Word):
            if parts and isinstance(prev, Article):
                parts.append(' ')
            elif (
                parts
                and isinstance(prev, Literal)
                and not prev.text.endswith(' ')
                and not token.text.startswith(' ')
            ):
                last_char = prev.text[-1] if prev.text else ''
                if last_char.isalnum():
                    parts.append(' ')
            parts.append(token.text)
        elif isinstance(token, Article):
            if parts:
                if isinstance(prev, (Word, Article)):
                    parts.append(' ')
                elif isinstance(prev, Literal) and not prev.text.endswith(' '):
                    parts.append(' ')
            parts.append(token.text)
        elif isinstance(token, Affix):
            # Affixes that were not attached by attach_affixes() are rendered as words.
            if parts and isinstance(prev, Article):
                parts.append(' ')
            parts.append(token.text)
        prev = token
    return ''.join(parts).strip()


#: Ordered token-stream assembly rules. Each rule takes and returns a token
#: list; rules run in declared order. Add or reorder rules here.
ASSEMBLY_RULES: list[Callable[[list[Token]], list[Token]]] = [
    apply_sticky,
    attach_affixes,
    apply_articles,
    title_case_tokens,
]


def render_pipeline(tokens: list[Token]) -> str:
    """Run all assembly rules in declared order and return the final string."""
    for rule in ASSEMBLY_RULES:
        tokens = rule(tokens)
    return render_tokens(tokens)


def title_case(text: str) -> str:
    """Public title-case wrapper for backwards compatibility."""
    words = text.split()
    if not words:
        return text
    result = []
    for i, word in enumerate(words):
        lowered = word.lower()
        if word in _PRESERVE_CASE:
            result.append(word)
        elif i == 0 or lowered not in _SMALL_WORDS:
            result.append(word[:1].upper() + word[1:])
        else:
            result.append(word)
    return ' '.join(result)
