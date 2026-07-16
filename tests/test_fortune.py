"""Tests for fortune / randomness tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import ara.fortune as fortune_grammar
from ara.world import fortune
from ara.world.orchestrator import Orchestrator

from tests.helpers import make_next_round_result as _make_next_round_result
from tests.helpers import make_scene as _make_scene


def test_load_hexagrams() -> None:
    hexagrams = fortune.load_hexagrams()
    assert len(hexagrams) == 64
    first = hexagrams[0]
    assert "number" in first
    assert "name" in first
    assert "chinese" in first
    assert "judgment" in first
    assert "image" in first
    assert "changing_lines" in first


def test_cast_iching_returns_valid_hexagram() -> None:
    h = fortune.cast_iching()
    assert h["chinese"]
    assert h["judgment"]
    # Moving lines are omitted by default to reduce ambiguity.
    assert "moving_lines" not in h


def test_cast_iching_verbose_includes_moving_lines() -> None:
    h = fortune.cast_iching(verbose=True)
    assert h["chinese"]
    assert h["judgment"]
    assert "moving_lines" in h
    # cast_iching returns a subset of hexagram fields, not 'number' or 'name'


def test_load_inspiration() -> None:
    words = fortune.load_inspiration()
    assert len(words) > 0
    assert all(isinstance(w, str) for w in words)


def test_random_inspiration() -> None:
    word = fortune.random_inspiration()
    assert isinstance(word, str)
    assert word in fortune.load_inspiration()


def test_supported_distributions() -> None:
    dists = fortune.supported_distributions()
    for name in ("uniform", "normal", "exponential", "erlang", "gamma", "beta",
                 "lognormal", "poisson", "binomial", "geometric", "pareto",
                 "weibull", "triangular", "laplace"):
        assert name in dists


def test_sample_uniform_is_in_range() -> None:
    for _ in range(10):
        v = fortune.sample_distribution("uniform")
        assert 0.0 <= v <= 1.0


def test_sample_normal_uses_params() -> None:
    # With mean=0 and small std, values should cluster near 0 (and be clamped to >=0).
    for _ in range(10):
        v = fortune.sample_distribution("normal", {"mean": 0.0, "std": 0.01})
        assert 0.0 <= v <= 1.0


def test_sample_discrete_distributions() -> None:
    for _ in range(10):
        poisson = fortune.sample_distribution("poisson", {"lam": 2.0})
        assert isinstance(poisson, int)
        assert poisson >= 0
        b = fortune.sample_distribution("binomial", {"n": 5, "p": 0.5})
        assert isinstance(b, int)
        assert 0 <= b <= 5
        g = fortune.sample_distribution("geometric", {"p": 0.5})
        assert isinstance(g, int)
        assert g >= 1


def test_sample_unknown_distribution_raises() -> None:
    with pytest.raises(ValueError):
        fortune.sample_distribution("not_a_dist")


def test_fortune_suite_has_all_fields() -> None:
    suite = fortune.fortune_suite()
    assert "roll" in suite
    assert "random" in suite
    assert "iching" in suite
    assert "inspiration" in suite
    # fortune_suite doesn't include 'title' - it's a separate tool


def test_list_title_flavors() -> None:
    flavors = fortune_grammar.list_title_flavors()
    assert isinstance(flavors, list)
    assert "generic" not in flavors
    assert "numbers" not in flavors
    assert "templates" not in flavors
    # A few expected flavors from the global title directory.
    assert "fantasy" in flavors
    assert "jrpg" in flavors
    assert "tfr" in flavors


def test_generate_title_returns_string() -> None:
    title = fortune_grammar.generate_title()
    assert isinstance(title, str)
    assert title


def test_generate_title_with_flavor() -> None:
    title = fortune_grammar.generate_title(flavors="fantasy")
    assert isinstance(title, str)
    assert title


def test_generate_title_with_level() -> None:
    title = fortune_grammar.generate_title(level="simple")
    assert isinstance(title, str)
    assert title


def test_orchestrator_registers_fortune_tools() -> None:
    mock_client = MagicMock()
    orch = Orchestrator(mock_client, db=None)

    # The orchestrator builds tools inside decide_next_turn, so we run a minimal
    # turn to populate the registry.
    mock_db = MagicMock(spec="ara.memory.chroma.ChromaStore")
    scene = _make_scene(
        "fortune_scene", mock_db, char_names=("Player", "Narrator", "Alice")
    )
    char = next(c for c in scene.character_pool if c.name == "Alice")
    ctx = MagicMock()
    ctx.branch.return_value = ctx
    ctx.head = None
    ctx.to_list.return_value = []

    mock_client.complete.return_value = _make_next_round_result("Alice")
    decision = orch.decide_next_turn(
        scene=scene,
        ctx=ctx,
        here_chars=scene.starting_characters,
        away_chars=set(),
        prev_char=None,
        loc=scene.starting_location,
    )
    assert decision.next_char == char
    assert "fortune_roll" in orch.registry
    assert "fortune_random" in orch.registry
    assert "fortune_iching" in orch.registry
    assert "fortune_inspiration" in orch.registry
    assert "fortune_suite" in orch.registry
    assert "fortune_title" in orch.registry
    assert "fortune_name" in orch.registry
    # Aliases should still work.
    assert "roll" in orch.registry
    assert "random" in orch.registry
    assert "title" in orch.registry
    assert "name" in orch.registry


def test_cull_grammar_caps_dominant_source() -> None:
    grammar = {
        '__generic_slots__': {'noun'},
        'noun': [
            {'value': 'a', '_source': 'x'},
            {'value': 'b', '_source': 'x'},
            {'value': 'c', '_source': 'x'},
            {'value': 'd', '_source': 'x'},
            {'value': 'e', '_source': 'x'},
            {'value': 'f', '_source': 'y'},
            {'value': 'g', '_source': 'z'},
        ],
    }
    culled = fortune_grammar.cull_grammar(grammar)
    # total 7 * 0.4 = 2.8 -> cap 2. source x (5) capped to 2, y and z (1 each) kept.
    assert len(culled['noun']) == 4
    assert len([e for e in culled['noun'] if e['_source'] == 'x']) == 2
    assert len([e for e in culled['noun'] if e['_source'] == 'y']) == 1
    assert len([e for e in culled['noun'] if e['_source'] == 'z']) == 1


def test_cull_grammar_is_random_per_call() -> None:
    grammar = {
        '__generic_slots__': {'noun'},
        'noun': [
            {'value': f'{i}', '_source': 'x'}
            for i in range(200)
        ]
        + [
            {'value': f'y{i}', '_source': 'y'}
            for i in range(50)
        ]
        + [
            {'value': f'z{i}', '_source': 'z'}
            for i in range(50)
        ],
    }
    first = {e['value'] for e in fortune_grammar.cull_grammar(grammar)['noun']}
    different = False
    for _ in range(10):
        second = {e['value'] for e in fortune_grammar.cull_grammar(grammar)['noun']}
        if second != first:
            different = True
            break
    assert different


class TestFortunePackageSurface:
    """The ara.fortune package re-exports its whole public API."""

    def test_all_exports_resolve(self) -> None:
        for name in fortune_grammar.__all__:
            assert hasattr(fortune_grammar, name), f"ara.fortune missing {name}"

    def test_tokens_module_exposed(self) -> None:
        assert fortune_grammar.tokens is not None
        assert callable(fortune_grammar.tokens.expand)


class TestResolveLevel:
    """Level parsing shared by title and ability generation."""

    def test_none_and_all_mean_no_level(self) -> None:
        assert fortune_grammar.resolve_level(None) == (None, False)
        assert fortune_grammar.resolve_level("all") == (None, False)

    def test_named_and_indexed_levels(self) -> None:
        assert fortune_grammar.resolve_level("simple") == ("simple", False)
        assert fortune_grammar.resolve_level(2) == ("complex", False)
        assert fortune_grammar.resolve_level("2") == ("complex", False)

    def test_bang_marks_exact(self) -> None:
        assert fortune_grammar.resolve_level("simple!") == ("simple", True)
        assert fortune_grammar.resolve_level("1!") == ("moderate", True)

    def test_invalid_levels_raise(self) -> None:
        with pytest.raises(ValueError):
            fortune_grammar.resolve_level(9)
        with pytest.raises(ValueError):
            fortune_grammar.resolve_level("bogus")


class TestAbilityGeneration:
    """Public ability-grammar API of the fortune package."""

    def test_list_ability_flavors(self) -> None:
        flavors = fortune_grammar.list_ability_flavors()
        assert isinstance(flavors, list)
        assert flavors == sorted(flavors)
        assert "generic" not in flavors
        assert "numbers" not in flavors
        assert "templates" not in flavors
        # A few expected flavors from the global ability directory.
        assert "fire" in flavors
        assert "melee" in flavors
        assert "corporate" in flavors

    def test_generate_ability_returns_string(self) -> None:
        ability = fortune_grammar.generate_ability()
        assert isinstance(ability, str)
        assert ability

    def test_generate_ability_with_flavor(self) -> None:
        ability = fortune_grammar.generate_ability(flavors="fire")
        assert isinstance(ability, str)
        assert ability

    def test_generate_ability_unknown_flavor_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown ability flavor"):
            fortune_grammar.generate_ability(flavors="no_such_flavor")


class TestCategorizedFlavors:
    """Flavors are grouped by the category key of their TOML files."""

    def test_categorized_title_flavors_cover_all_flavors(self) -> None:
        categorized = fortune_grammar.categorized_title_flavors()
        assert categorized
        grouped = {f for flavors in categorized.values() for f in flavors}
        assert grouped == set(fortune_grammar.list_title_flavors())
        for flavors in categorized.values():
            assert flavors == sorted(flavors)

    def test_categorized_ability_flavors_cover_all_flavors(self) -> None:
        categorized = fortune_grammar.categorized_ability_flavors()
        assert categorized
        grouped = {f for flavors in categorized.values() for f in flavors}
        assert grouped == set(fortune_grammar.list_ability_flavors())
        for flavors in categorized.values():
            assert flavors == sorted(flavors)


class TestGenerateName:
    """Human-name generation from the bundled names dataset."""

    def test_generate_name_default(self) -> None:
        name = fortune_grammar.generate_name()
        assert isinstance(name, str)
        assert len(name.split()) >= 2

    def test_generate_name_simple_style_has_two_parts(self) -> None:
        assert len(fortune_grammar.generate_name(style="simple").split()) == 2

    def test_generate_name_middle_style_has_three_parts(self) -> None:
        assert len(fortune_grammar.generate_name(style="middle").split()) == 3

    def test_generate_name_n_parts_overrides_style(self) -> None:
        assert len(fortune_grammar.generate_name(style="simple", n_parts=4).split()) == 4
