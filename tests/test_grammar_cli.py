"""Tests for shared grammar CLI helpers."""

from __future__ import annotations

import pytest

from scripts._grammar_cli import filter_templates, parse_require_args


def test_require_base_slot_matches_modifiers() -> None:
    templates = [
        "{verb}",
        "{verb:noun}",
        "{verb:gerund} {noun}",
        "{noun} of {place}",
    ]
    result = filter_templates(templates, ["verb"])
    assert result == [
        "{verb}",
        "{verb:noun}",
        "{verb:gerund} {noun}",
    ]


def test_require_exact_bare_slot_with_at() -> None:
    templates = [
        "{verb}",
        "{verb:noun}",
        "{noun}",
    ]
    result = filter_templates(templates, ["@verb"])
    assert result == ["{verb}"]


def test_require_and() -> None:
    templates = [
        "{verb} {noun}",
        "{verb}",
        "{noun}",
    ]
    result = filter_templates(templates, ["verb,noun"])
    assert result == ["{verb} {noun}"]


def test_require_or() -> None:
    templates = [
        "{verb}",
        "{noun}",
        "{place}",
    ]
    result = filter_templates(templates, ["verb+noun"])
    assert result == ["{verb}", "{noun}"]


def test_require_prefix_plus_matches_prefix() -> None:
    templates = [
        "{prefix} {noun}",
        "{prefix+}{noun}",
        "{noun}",
    ]
    result = filter_templates(templates, ["prefix"])
    assert result == [
        "{prefix} {noun}",
        "{prefix+}{noun}",
    ]


def test_require_no_match_exits(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        filter_templates(["{noun}"], ["verb"])
    captured = capsys.readouterr()
    assert "No templates match required slots: verb" in captured.out


def test_parse_require_args_simple_alias() -> None:
    cleaned, sources = parse_require_args(["verb:melee"])
    assert cleaned == ["verb"]
    assert sources == {"verb": [["melee"]]}


def test_parse_require_args_complex_alias() -> None:
    cleaned, sources = parse_require_args(["foo:@bar:baz+a:b"])
    assert cleaned == ["foo+a"]
    assert sources == {
        "foo": [["@bar:baz"]],
        "a": [["b"]],
    }


def test_parse_require_args_exact_with_sources() -> None:
    cleaned, sources = parse_require_args(["@verb:melee"])
    assert cleaned == ["@verb"]
    assert sources == {"verb": [["melee"]]}


def test_parse_require_args_multiple_expressions() -> None:
    cleaned, sources = parse_require_args(["verb:melee", "noun:fire"])
    assert cleaned == ["verb", "noun"]
    assert sources == {
        "verb": [["melee"]],
        "noun": [["fire"]],
    }


def test_parse_require_args_empty() -> None:
    assert parse_require_args(None) == ([], {})
    assert parse_require_args([]) == ([], {})
