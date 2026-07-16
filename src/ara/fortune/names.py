"""Human name generation from the humannames dataset.

Loads ~200K first/last names from FinNLP/humannames and generates random
given+surname combinations with optional middle names or long-form names.
"""

from __future__ import annotations

import random
from pathlib import Path

from ara.config import AraSettings

_NAMES_CACHE: list[str] | None = None


def _data_dir(config: AraSettings | None = None) -> Path:
    config = config or AraSettings()
    return config.fortune_path(None) / "names"


def _load_names(config: AraSettings | None = None) -> list[str]:
    global _NAMES_CACHE
    if _NAMES_CACHE is None:
        path = _data_dir(config) / "list.txt"
        with path.open() as f:
            _NAMES_CACHE = [line.strip() for line in f if line.strip()]
    return _NAMES_CACHE


def generate_name(
    style: str = "random",
    n_parts: int | None = None,
    config: AraSettings | None = None,
) -> str:
    """Generate a random human name by combining name parts.

    Parameters
    ----------
    style:
        - ``"random"`` (default): weighted distribution favoring 2-3 parts.
        - ``"simple"``: first + last (2 parts).
        - ``"middle"``: first + middle + last (3 parts).
        - ``"spanish"``: 4-6 parts, Spanish-style long form.
    n_parts:
        Override the number of parts.  Takes precedence over *style*.
    config:
        Optional settings instance (defaults to a fresh :class:`AraSettings`).

    Returns
    -------
    str
        A space-joined human name, e.g. ``"Aadhya Corvi"`` or
        ``"Jose Maria Garcia Lopez de la Cruz"``.
    """
    names = _load_names(config)

    if n_parts is not None:
        count = max(1, n_parts)
    elif style == "simple":
        count = 2
    elif style == "middle":
        count = 3
    elif style == "spanish":
        count = random.randint(4, 6)
    else:
        # Weighted distribution: 60% 2-part, 25% 3-part, 10% 4-part, 5% 5-6 part
        weights = [0, 0, 60, 25, 10, 3, 2]
        count = random.choices(range(len(weights)), weights=weights, k=1)[0]

    parts = [random.choice(names) for _ in range(count)]
    return " ".join(parts)
