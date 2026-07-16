"""Tests for :mod:`ara.agent.state`."""

from __future__ import annotations

from unittest.mock import MagicMock

from ara.agent.state import engine_to_dict
from ara.memory.chroma import ChromaStore
from ara.world.engine import Engine

from tests.helpers import make_scene


def test_character_statuses_have_default_title() -> None:
    """Empty character statuses should still expose the default page title."""
    mock_db = MagicMock(spec=ChromaStore)
    scene = make_scene(mock_db=mock_db)
    engine = Engine(MagicMock(), db=mock_db)  # type: ignore[arg-type]
    engine.start(scene)

    state = engine_to_dict(engine)

    assert "character_statuses" in state
    for name, status in state["character_statuses"].items():
        assert status.get("title") == "Status", f"{name} status missing default title"


def test_location_statuses_have_default_title() -> None:
    """Empty location statuses should still expose the default page title."""
    mock_db = MagicMock(spec=ChromaStore)
    scene = make_scene(mock_db=mock_db)
    engine = Engine(MagicMock(), db=mock_db)  # type: ignore[arg-type]
    engine.start(scene)

    state = engine_to_dict(engine)

    assert "location_statuses" in state
    for name, status in state["location_statuses"].items():
        assert status.get("title") == "Status", f"{name} status missing default title"
