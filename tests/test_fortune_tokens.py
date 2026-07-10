import pytest

from ara.world import _fortune_tokens as tokens


def test_tokenize_literal() -> None:
    assert tokens.tokenize("hello world") == [tokens.Literal("hello world")]


def test_tokenize_simple_slot() -> None:
    assert tokens.tokenize("{noun}") == [tokens.Slot("noun", ())]


def test_tokenize_slot_with_modifiers() -> None:
    assert tokens.tokenize("{verb:noun:plural}") == [
        tokens.Slot("verb", ("noun", "plural"))
    ]


def test_tokenize_mixed() -> None:
    assert tokens.tokenize("the {adj} {noun:plural}") == [
        tokens.Literal("the "),
        tokens.Slot("adj", ()),
        tokens.Literal(" "),
        tokens.Slot("noun", ("plural",)),
    ]


def test_expand_literal() -> None:
    assert tokens.expand("hello world", {}) == "hello world"


def test_expand_single_slot() -> None:
    grammar = {"noun": [{"value": "king"}]}
    assert tokens.expand("{noun}", grammar) == "King"


def test_expand_prefix_suffix() -> None:
    grammar = {
        "prefix": [{"value": "battle"}],
        "suffix": [{"value": "mage"}],
    }
    assert tokens.expand("{prefix}{suffix}", grammar) == "Battlemage"


def test_prefix_suffix_attachment() -> None:
    grammar = {
        "prefix": [{"value": "fire"}],
        "suffix": [{"value": "walk"}],
    }
    assert tokens.expand("{prefix}{suffix}", grammar) == "Firewalk"


def test_article_drop_after_preposition() -> None:
    # "of the" should keep lowercase "the" even after title casing
    grammar = {
        "title": [{"value": "the lord of the rings"}],
    }
    assert tokens.expand("{title}", grammar) == "The Lord of the Rings"


def test_plural_modifiers() -> None:
    grammar = {"noun": [{"value": "wolf"}]}
    assert tokens.expand("{noun:plural}", grammar) == "Wolves"


def test_gerund_modifier() -> None:
    grammar = {"verb": [{"value": "run"}]}
    assert tokens.expand("{verb:gerund}", grammar) == "Running"


def test_past_modifier() -> None:
    grammar = {"verb": [{"value": "walk"}]}
    assert tokens.expand("{verb:past}", grammar) == "Walked"


def test_noun_form_modifier() -> None:
    grammar = {"verb": [{"value": "strike", "noun_form": "strike"}]}
    assert tokens.expand("{verb:noun}", grammar) == "Strike"


def test_noun_modifier_falls_back_to_base() -> None:
    grammar = {"verb": [{"value": "run"}]}
    # Falls back to the base value when noun_form is absent.
    assert tokens.expand("{verb:noun}", grammar) == "Run"


def test_pos_noun_override() -> None:
    grammar = {"noun": [{"value": "meteor", "pos": "noun"}]}
    # pos=noun means :noun leaves the value unchanged.
    assert tokens.expand("{noun:noun}", grammar) == "Meteor"


def test_value_with_embedded_slot() -> None:
    grammar = {
        "digit": [{"value": "7"}],
        "noun": [{"value": "A({digit})"}],
    }
    assert tokens.expand("{noun}", grammar) == "A(7)"


def test_title_case_wrapper() -> None:
    assert tokens.title_case("the grim knight of the shadowfell") == "The Grim Knight of the Shadowfell"


def test_expand_all_enumerates() -> None:
    grammar = {
        "adj": [{"value": "grim"}, {"value": "old"}],
        "suffix": [{"value": "knight"}, {"value": "mancy"}],
    }
    results = tokens.expand_all("{adj}{suffix}", grammar)
    assert sorted(results) == sorted(
        ["Grimknight", "Grimmancy", "Oldknight", "Oldmancy"]
    )


def test_modifier_propagation_through_pattern_slot() -> None:
    """Modifiers on a pattern-style slot should flow to its single-slot pattern."""
    grammar = {
        "technique": [{"patterns": ["{verb}"]}],
        "verb": [{"value": "strike", "noun_form": "strike"}],
    }
    assert tokens.expand("{technique:noun}", grammar) == "Strike"


def test_modifier_propagation_through_pattern_slot_all() -> None:
    """expand_all should also propagate modifiers through single-slot patterns."""
    grammar = {
        "technique": [{"patterns": ["{verb}"]}],
        "verb": [
            {"value": "strike", "noun_form": "strike"},
            {"value": "burn", "noun_form": "burn"},
        ],
    }
    results = tokens.expand_all("{technique:noun}", grammar)
    assert sorted(results) == sorted(["Strike", "Burn"])


def test_multi_slot_pattern_does_not_propagate_modifiers() -> None:
    """Patterns with more than one slot should not inherit modifiers."""
    grammar = {
        "phrase": [{"patterns": ["{prefix}{suffix}"]}],
        "prefix": [{"value": "battle"}],
        "suffix": [{"value": "mage"}],
    }
    # The :noun modifier applies only to the [[phrase]] entry itself, which has
    # no noun_form, so it should be a no-op and expand normally.
    assert tokens.expand("{phrase:noun}", grammar) == "Battlemage"


def test_traced_expansion_records_parent() -> None:
    grammar = {
        "title": [{"patterns": ["{adj} {noun}"]}],
        "adj": [{"value": "grim"}],
        "noun": [{"value": "knight"}],
    }
    result, trace = tokens.expand_traced("{title}", grammar)
    assert result == "Grim Knight"
    parents = {entry.get("slot"): entry.get("parent") for entry in trace}
    assert parents.get("adj") == "title"
    assert parents.get("noun") == "title"


def test_selects_prefixable_entry_when_prefix_adjacent() -> None:
    grammar = {
        "prefix": [{"value": "neo"}],
        "noun": [
            {"value": "one", "prefixable": True, "suffixible": True},
            {"value": "world", "prefixable": False, "suffixible": False},
        ],
    }
    for _ in range(20):
        assert tokens.expand("{prefix}{noun}", grammar) == "Neoone"


def test_selects_suffixible_entry_when_suffix_adjacent() -> None:
    grammar = {
        "noun": [
            {"value": "one", "prefixable": True, "suffixible": True},
            {"value": "world", "prefixable": False, "suffixible": False},
        ],
        "suffix": [{"value": "ling"}],
    }
    for _ in range(20):
        assert tokens.expand("{noun}{suffix}", grammar) == "Oneling"


def test_selects_both_flags_for_prefix_noun_suffix() -> None:
    grammar = {
        "prefix": [{"value": "neo"}],
        "noun": [
            {"value": "one", "prefixable": True, "suffixible": True},
            {"value": "world", "prefixable": False, "suffixible": False},
        ],
        "suffix": [{"value": "ling"}],
    }
    for _ in range(20):
        assert tokens.expand("{prefix}{noun}{suffix}", grammar) == "Neooneling"


def test_drops_suffix_when_no_suffixible_entry() -> None:
    grammar = {
        "noun": [{"value": "world", "prefixable": False, "suffixible": False}],
        "suffix": [{"value": "ling"}],
    }
    assert tokens.expand("{noun}{suffix}", grammar) == "World"


def test_drops_prefix_when_no_prefixable_entry() -> None:
    grammar = {
        "prefix": [{"value": "neo"}],
        "noun": [{"value": "world", "prefixable": False, "suffixible": False}],
    }
    assert tokens.expand("{prefix}{noun}", grammar) == "World"


def test_expand_all_respects_affinity_flags() -> None:
    grammar = {
        "prefix": [{"value": "neo"}],
        "noun": [
            {"value": "one", "prefixable": True},
            {"value": "world", "prefixable": False},
        ],
    }
    results = tokens.expand_all("{prefix}{noun}", grammar)
    assert sorted(results) == sorted(["Neoone"])


def test_group_reuses_same_entry_with_modifiers() -> None:
    grammar = {
        "verb": [
            {"value": "strike", "noun_form": "strike", "gerund": "striking", "plural": "strikes"},
            {"value": "burn", "noun_form": "burn", "gerund": "burning", "plural": "burns"},
        ],
    }
    for _ in range(20):
        result = tokens.expand("{verb:gerund;0} {verb;0} {verb:plural;0}", grammar)
        parts = result.split()
        assert len(parts) == 3
        base = parts[1]
        assert parts[0] == tokens.default_gerund(base)
        assert parts[2] == tokens.default_plural(base)


def test_group_same_slot_different_groups() -> None:
    grammar = {
        "foo": [{"value": "alpha"}, {"value": "beta"}, {"value": "gamma"}],
    }
    seen: set[str] = set()
    for _ in range(40):
        result = tokens.expand("{foo;0} {foo;0} {foo;1}", grammar)
        seen.add(result)
    # First two words are always identical; third word can differ.
    for outcome in seen:
        parts = outcome.split()
        assert parts[0] == parts[1]
        assert len(parts) == 3


def test_group_same_slot_same_group_repeats() -> None:
    grammar = {
        "foo": [{"value": "alpha"}, {"value": "beta"}],
    }
    for _ in range(20):
        result = tokens.expand("{foo;0} {foo;0} {foo;1} {foo;1}", grammar)
        a, b, c, d = result.split()
        assert a == b
        assert c == d


def test_group_disjoint_across_slots() -> None:
    grammar = {
        "foo": [{"value": "alpha"}, {"value": "beta"}],
        "bar": [{"value": "alpha"}, {"value": "beta"}],
    }
    for _ in range(20):
        result = tokens.expand("{foo;GRP} {bar;GRP}", grammar)
        a, b = result.split()
        assert a != b


def test_group_on_composite_slot_ignored() -> None:
    grammar = {
        "prefix+": [{"patterns": ["{prefix}"]}],
        "prefix": [{"value": "neo"}, {"value": "meta"}],
        "noun": [{"value": "one"}],
    }
    results = tokens.expand_all("{prefix+;GRP}{noun}", grammar)
    assert sorted(results) == sorted(["Neoone", "Metaone"])


def test_group_composite_reuses_literal_expansion() -> None:
    grammar = {
        "sub": [
            {"patterns": ["{a}"]},
            {"patterns": ["{a}-{b}"]},
        ],
        "a": [{"value": "x"}, {"value": "y"}],
        "b": [{"value": "1"}, {"value": "2"}],
    }
    for _ in range(30):
        result = tokens.expand("{sub;G} {sub;G}", grammar)
        first, second = result.split()
        # A grouped composite slot records and reuses its literal expansion.
        assert first == second


def test_composite_modifier_forwarded_to_last_slot() -> None:
    grammar = {
        "foo": [{"patterns": ["{class}-{noun}"]}],
        "class": [{"value": "dark"}],
        "noun": [{"value": "walker", "plural": "walkers"}],
    }
    assert tokens.expand("{foo:plural}", grammar) == "Dark-Walkers"


def test_composite_group_reuses_inner_entries_with_modifiers() -> None:
    grammar = {
        "foo": [{"patterns": ["{class}-{noun}"]}],
        "class": [{"value": "dark"}, {"value": "grim"}],
        "noun": [
            {"value": "walker", "plural": "walkers"},
            {"value": "being", "plural": "beings"},
        ],
    }
    for _ in range(20):
        result = tokens.expand("{foo;G} {foo:plural;G}", grammar)
        base, plural = result.split()
        base_word = base.split("-")[1]
        plural_word = plural.split("-")[1]
        assert base.split("-")[0] == plural.split("-")[0]
        assert plural_word == tokens.default_plural(base_word)


def test_composite_group_reuses_pattern_shape() -> None:
    grammar = {
        "foo": [
            {"patterns": ["{a}"]},
            {"patterns": ["{a}-{b}"]},
        ],
        "a": [{"value": "x"}, {"value": "y"}],
        "b": [{"value": "1"}, {"value": "2"}],
    }
    for _ in range(30):
        result = tokens.expand("{foo;G} {foo;G}", grammar)
        first, second = result.split()
        assert first == second


def test_composite_single_slot_modifier_propagation_still_works() -> None:
    grammar = {
        "phrase": [{"patterns": ["{verb}"]}],
        "verb": [{"value": "strike", "noun_form": "strike"}],
    }
    assert tokens.expand("{phrase:noun}", grammar) == "Strike"


def test_lowercase_meta_preserved_at_start() -> None:
    grammar = {
        "word": [{"value": "tty", "lowercase": True}],
    }
    assert tokens.expand("{word}", grammar) == "tty"


def test_lowercase_meta_preserved_in_middle() -> None:
    grammar = {
        "word": [{"value": "tty", "lowercase": True}],
        "other": [{"value": "device"}],
    }
    assert tokens.expand("{other} {word}", grammar) == "Device tty"


def test_lowercase_false_or_absent_still_title_cased() -> None:
    grammar = {
        "word": [{"value": "tty"}],
    }
    assert tokens.expand("{word}", grammar) == "Tty"


def test_adj_sup_not_lowercased_when_not_leading() -> None:
    grammar = {
        "adj_sup": [{"value": "Yet another"}],
        "noun": [{"value": "thing"}],
    }
    assert tokens.expand("{noun} {adj_sup}", grammar) == "Thing Yet Another"


# --------------------------------------------------------------------------- #
# Sticky metadata
# --------------------------------------------------------------------------- #


def test_apply_sticky_inserts_space_for_non_sticky_word() -> None:
    sticky = tokens.Word('foo', {'sticky': True})
    non_sticky = tokens.Word('bar', {'sticky': False})
    assert tokens.apply_sticky([sticky, non_sticky]) == [
        sticky,
        tokens.Literal(' '),
        non_sticky,
    ]


def test_apply_sticky_space_when_non_sticky_leads() -> None:
    sticky = tokens.Word('foo', {'sticky': True})
    non_sticky = tokens.Word('bar', {'sticky': False})
    assert tokens.apply_sticky([non_sticky, sticky]) == [
        non_sticky,
        tokens.Literal(' '),
        sticky,
    ]


def test_apply_sticky_replaces_hyphen_for_non_sticky() -> None:
    sticky = tokens.Word('foo', {'sticky': True})
    non_sticky = tokens.Word('bar', {'sticky': False})
    assert tokens.apply_sticky([sticky, tokens.Literal('-'), non_sticky]) == [
        sticky,
        tokens.Literal(' '),
        non_sticky,
    ]


def test_apply_sticky_preserves_comma_and_colon_separators() -> None:
    sticky = tokens.Word('foo', {'sticky': True})
    non_sticky = tokens.Word('bar', {'sticky': False})
    assert tokens.apply_sticky([sticky, tokens.Literal(', '), non_sticky]) == [
        sticky,
        tokens.Literal(', '),
        non_sticky,
    ]
    assert tokens.apply_sticky([sticky, tokens.Literal(': '), non_sticky]) == [
        sticky,
        tokens.Literal(': '),
        non_sticky,
    ]


def test_apply_sticky_leaves_sticky_pair_untouched() -> None:
    a = tokens.Word('foo', {'sticky': True})
    b = tokens.Word('bar', {'sticky': True})
    assert tokens.apply_sticky([a, b]) == [a, b]


def test_apply_sticky_hyphen_preserved_when_both_sticky() -> None:
    a = tokens.Word('foo', {'sticky': True})
    b = tokens.Word('bar', {'sticky': True})
    assert tokens.apply_sticky([a, tokens.Literal('-'), b]) == [
        a,
        tokens.Literal('-'),
        b,
    ]


def test_expand_non_sticky_word_forces_space() -> None:
    grammar = {
        'foo': [{'value': 'foo', 'sticky': False}],
        'bar': [{'value': 'bar'}],
    }
    assert tokens.expand('{foo}{bar}', grammar) == 'Foo Bar'
    assert tokens.expand('{bar}{foo}', grammar) == 'Bar Foo'


def test_expand_non_sticky_word_replaces_hyphen() -> None:
    grammar = {
        'foo': [{'value': 'foo', 'sticky': False}],
        'bar': [{'value': 'bar'}],
    }
    assert tokens.expand('{foo}-{bar}', grammar) == 'Foo Bar'


def test_expand_non_sticky_word_keeps_comma_colon() -> None:
    grammar = {
        'foo': [{'value': 'foo', 'sticky': False}],
        'bar': [{'value': 'bar'}],
    }
    assert tokens.expand('{foo}, {bar}', grammar) == 'Foo, Bar'
    assert tokens.expand('{foo}: {bar}', grammar) == 'Foo: Bar'


def test_expand_non_sticky_word_blocks_affix_attachment() -> None:
    grammar = {
        'prefix': [{'value': 'neo'}],
        'suffix': [{'value': 'ling'}],
        'noun': [{'value': 'sous chef', 'sticky': False}],
    }
    assert tokens.expand('{prefix}{noun}', grammar) == 'Neo Sous Chef'
    assert tokens.expand('{noun}{suffix}', grammar) == 'Sous Chef'


def test_expand_sticky_word_allows_affix_attachment() -> None:
    grammar = {
        'prefix': [{'value': 'neo'}],
        'noun': [{'value': 'chef'}],
    }
    assert tokens.expand('{prefix}{noun}', grammar) == 'Neochef'


def test_supersticky_prefix_sticks_right() -> None:
    grammar = {
        'prefix': [{'value': 'GNU/', 'supersticky': True}],
        'noun': [{'value': 'distro hopper', 'sticky': False}],
    }
    assert tokens.expand('{prefix}{noun}', grammar) == 'GNU/Distro Hopper'


def test_supersticky_plus_prefix_sticks_right() -> None:
    grammar = {
        'prefix': [{'value': 'GNU+', 'supersticky': True}],
        'noun': [{'value': 'linux', 'sticky': False}],
    }
    assert tokens.expand('{prefix}{noun}', grammar) == 'GNU+Linux'


def test_suffix_still_attaches_when_word_is_sticky() -> None:
    grammar = {
        'suffix': [{'value': 'ling'}],
        'noun': [{'value': 'sous'}],
    }
    assert tokens.expand('{noun}{suffix}', grammar) == 'Sousling'


def test_suffix_drops_when_word_rejects_it() -> None:
    grammar = {
        'suffix': [{'value': 'ling'}],
        'noun': [{'value': 'sous', 'suffixible': False}],
    }
    assert tokens.expand('{noun}{suffix}', grammar) == 'Sous'
