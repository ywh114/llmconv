"""Randomness and fortune tools for the orchestrator.

These functions provide true randomness and curated random flavor that LLMs are
poor at generating on their own. The orchestrator calls them and interprets the
results; they do not automatically resolve scenes.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from ara.config import AraSettings
from ara.world import ability as _ability
from ara.world import title as _title
from ara.world import name as _name


# --------------------------------------------------------------------------- #
# I-Ching
# --------------------------------------------------------------------------- #

def _data_dir(story: str | None = None, config: AraSettings | None = None) -> Path:
    settings = config or AraSettings()
    return settings.fortune_path(story)


def load_hexagrams(story: str | None = None, config: AraSettings | None = None) -> list[dict[str, Any]]:
    """Load the 64 I-Ching hexagrams.

    Per-story files take priority over global files.
    """
    if story:
        path = _data_dir(story, config) / "iching.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    path = _data_dir(None, config) / "iching.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cast_iching(
    story: str | None = None,
    config: AraSettings | None = None,
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Cast an I-Ching hexagram.

    :param story: Optional story name for per-story hexagram data.
    :param config: Optional settings override.
    :param verbose: If ``True``, include randomly-selected moving/changing lines.
    :return: Dict with ``chinese``, ``judgment``, and optionally ``moving_lines``.
    """
    hexagrams = load_hexagrams(story, config=config)
    if not hexagrams:
        return {"chinese": "", "judgment": ""}

    hexagram = random.choice(hexagrams)

    result: dict[str, Any] = {
        "chinese": hexagram.get("chinese", ""),
        "judgment": hexagram.get("judgment", ""),
    }

    if verbose:
        changing_lines = hexagram.get("changing_lines", [])
        # Randomly select 0-3 moving lines
        moving_count = random.choice([0, 1, 1, 2, 2, 3])
        if moving_count and moving_count <= len(changing_lines):
            moving_lines = random.sample(changing_lines, moving_count)
        else:
            moving_lines = []
        result["moving_lines"] = moving_lines

    return result


# --------------------------------------------------------------------------- #
# Random inspiration
# --------------------------------------------------------------------------- #

def load_inspiration(story: str | None = None, config: AraSettings | None = None) -> list[str]:
    """Load the inspiration word list.

    Per-story files take priority over global files.
    """
    if story:
        path = _data_dir(story, config) / "inspiration.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    path = _data_dir(None, config) / "inspiration.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def random_inspiration(story: str | None = None, config: AraSettings | None = None) -> str:
    """Return a random inspiration word or phrase."""
    words = load_inspiration(story, config=config)
    if not words:
        return "silence"
    return random.choice(words)


# --------------------------------------------------------------------------- #
# Distribution sampling
# --------------------------------------------------------------------------- #

_DISTRIBUTIONS = {
    "uniform",
    "normal",
    "exponential",
    "erlang",
    "gamma",
    "beta",
    "lognormal",
    "poisson",
    "binomial",
    "geometric",
    "pareto",
    "weibull",
    "triangular",
    "laplace",
}


def supported_distributions() -> set[str]:
    """Return the set of supported distribution names."""
    return set(_DISTRIBUTIONS)


def sample_distribution(distrib: str, params: dict[str, Any] | None = None) -> float | int:
    """Sample a value from the named distribution.

    :param distrib: Distribution name (e.g. ``normal``, ``exponential``).
    :param params: Distribution-specific parameters.
    :return: Sampled value (float for continuous, int for discrete).
    :raises ValueError: If the distribution is unknown.
    """
    params = params or {}
    if distrib not in _DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution '{distrib}'.")

    if distrib == "uniform":
        return random.random()

    if distrib == "normal":
        mean = float(params.get("mean", 0.5))
        std = float(params.get("std", 0.15))
        return max(0.0, min(1.0, random.gauss(mean, std)))

    if distrib == "exponential":
        rate = float(params.get("rate", 1.0))
        if rate <= 0:
            raise ValueError("exponential rate must be positive")
        return random.expovariate(rate)

    if distrib == "erlang":
        shape = int(params.get("shape", 1))
        scale = float(params.get("scale", 1.0))
        if shape <= 0 or scale <= 0:
            raise ValueError("erlang shape and scale must be positive")
        return sum(random.expovariate(1.0 / scale) for _ in range(shape))

    if distrib == "gamma":
        shape = float(params.get("shape", 1.0))
        scale = float(params.get("scale", 1.0))
        if shape <= 0 or scale <= 0:
            raise ValueError("gamma shape and scale must be positive")
        return random.gammavariate(shape, scale)

    if distrib == "beta":
        alpha = float(params.get("alpha", 1.0))
        beta = float(params.get("beta", 1.0))
        if alpha <= 0 or beta <= 0:
            raise ValueError("beta alpha and beta must be positive")
        return random.betavariate(alpha, beta)

    if distrib == "lognormal":
        mean = float(params.get("mean", 0.0))
        sigma = float(params.get("sigma", 1.0))
        if sigma <= 0:
            raise ValueError("lognormal sigma must be positive")
        return random.lognormvariate(mean, sigma)

    if distrib == "poisson":
        lam = float(params.get("lam", 1.0))
        if lam <= 0:
            raise ValueError("poisson lam must be positive")
        # Knuth's method for small lambda
        if lam < 30.0:
            l = math.exp(-lam)
            k = 0
            p = 1.0
            while p > l:
                k += 1
                p *= random.random()
            return k - 1
        # Normal approximation for large lambda
        return int(max(0.0, random.gauss(lam, math.sqrt(lam))))

    if distrib == "binomial":
        n = int(params.get("n", 1))
        p = float(params.get("p", 0.5))
        if n < 0:
            raise ValueError("binomial n must be non-negative")
        if not 0.0 <= p <= 1.0:
            raise ValueError("binomial p must be in [0, 1]")
        return sum(random.random() < p for _ in range(n))

    if distrib == "geometric":
        p = float(params.get("p", 0.5))
        if not 0.0 < p <= 1.0:
            raise ValueError("geometric p must be in (0, 1]")
        return math.ceil(math.log(1.0 - random.random()) / math.log(1.0 - p))

    if distrib == "pareto":
        alpha = float(params.get("alpha", 1.0))
        if alpha <= 0:
            raise ValueError("pareto alpha must be positive")
        return random.paretovariate(alpha)

    if distrib == "weibull":
        shape = float(params.get("shape", 1.0))
        scale = float(params.get("scale", 1.0))
        if shape <= 0 or scale <= 0:
            raise ValueError("weibull shape and scale must be positive")
        return scale * random.weibullvariate(1.0, shape)

    if distrib == "triangular":
        low = float(params.get("low", 0.0))
        high = float(params.get("high", 1.0))
        mode = float(params.get("mode", (low + high) / 2))
        return random.triangular(low, high, mode)

    if distrib == "laplace":
        mean = float(params.get("mean", 0.0))
        scale = float(params.get("scale", 1.0))
        if scale <= 0:
            raise ValueError("laplace scale must be positive")
        u = random.random() - 0.5
        return mean - scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))

    raise ValueError(f"Unknown distribution '{distrib}'.")


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #

def list_title_flavors(
    story: str | None = None, config: AraSettings | None = None
) -> list[str]:
    """Return the available title grammar flavor names."""
    return _title.list_title_flavors(story, config)


def categorized_title_flavors(
    story: str | None = None, config: AraSettings | None = None
) -> dict[str, list[str]]:
    """Return title flavors grouped by category."""
    return _title.categorized_title_flavors(story, config)


def cull_grammar(grammar: dict) -> dict:
    """Return a source-capped copy of a title or ability grammar."""
    return _title.cull_grammar(grammar)


def load_title_grammar(
    story: str | None = None,
    config: AraSettings | None = None,
    flavors: list[str] | str | None = None,
    slot_sources: dict[str, list[str]] | None = None,
    cull_sources: bool = True,
) -> dict:
    """Load and merge title flavor grammars.

    Per-story title files take priority over global files.
    """
    return _title.load_title_grammar(
        story, config, flavors, slot_sources, cull_sources=cull_sources
    )


def generate_title(
    story: str | None = None,
    config: AraSettings | None = None,
    template: str | None = None,
    flavors: list[str] | str | None = None,
    level: str | int | None = "2",
    slot_sources: dict[str, list[str]] | None = None,
    required_slots: list[str] | set[str] | None = None,
    cull_sources: bool = True,
) -> str:
    """Generate a random title from the title grammar."""
    return _title.generate_title(
        story,
        config,
        template,
        flavors,
        level,
        slot_sources,
        required_slots,
        cull_sources=cull_sources,
    )


def fortune_suite(story: str | None = None, config: AraSettings | None = None) -> dict[str, Any]:
    """Return several independent random values at once.

    The orchestrator may use any or all of these as inputs.
    """
    return {
        "roll": f"Rolled 1d100: {random.randint(1, 100)}",
        "random": f"Random normal value: {sample_distribution('normal'):.4f}",
        "iching": cast_iching(story, config=config),
        "inspiration": random_inspiration(story, config=config),
    }


# --------------------------------------------------------------------------- #
# Ability generation
# --------------------------------------------------------------------------- #

def list_ability_flavors(
    story: str | None = None, config: AraSettings | None = None
) -> list[str]:
    """Return the available ability grammar flavor names."""
    return _ability.list_ability_flavors(story, config)


def categorized_ability_flavors(
    story: str | None = None, config: AraSettings | None = None
) -> dict[str, list[str]]:
    """Return ability flavors grouped by category."""
    return _ability.categorized_ability_flavors(story, config)


def load_ability_grammar(
    story: str | None = None,
    config: AraSettings | None = None,
    flavors: list[str] | str | None = None,
    slot_sources: dict[str, list[str]] | None = None,
    cull_sources: bool = True,
) -> dict:
    """Load and merge ability flavor grammars."""
    return _ability.load_ability_grammar(
        story, config, flavors, slot_sources, cull_sources=cull_sources
    )


def generate_ability(
    story: str | None = None,
    config: AraSettings | None = None,
    template: str | None = None,
    flavors: list[str] | str | None = None,
    level: str | int | None = "2",
    slot_sources: dict[str, list[str]] | None = None,
    required_slots: list[str] | set[str] | None = None,
    cull_sources: bool = True,
) -> str:
    """Generate a random ability name from the ability grammar."""
    return _ability.generate_ability(
        story,
        config,
        template,
        flavors,
        level,
        slot_sources,
        required_slots,
        cull_sources=cull_sources,
    )


def generate_name(
    style: str = "random",
    n_parts: int | None = None,
) -> str:
    """Generate a random human name from the humannames dataset."""
    return _name.generate_name(style=style, n_parts=n_parts)
