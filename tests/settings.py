"""Shared test settings for all Ara unit tests.

Import ``TEST_SETTINGS`` when you need an :class:`~ara.config.AraSettings`
instance inside a test.  It uses a dedicated ``data/tests/`` directory so
that leaked artifacts never pollute the main ``data/`` tree.
"""

from __future__ import annotations

from pathlib import Path

from ara.config import AraSettings

TEST_SETTINGS = AraSettings(data_dir=Path("data/tests"))
