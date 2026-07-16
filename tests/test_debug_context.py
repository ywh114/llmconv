"""Tests for the character-context debug command."""

from __future__ import annotations

from unittest.mock import MagicMock

from ara.agent.debug_bridge import StructuredDebugBridge
from ara.llm.client import LLMClient
from ara.world.engine import Engine

from tests.helpers import make_scene


def _make_engine_and_state(scene_id: str = "debug_scene"):
    """Create a started engine and a debug state for it."""
    mock_db = MagicMock()
    scene = make_scene(
        scene_id, mock_db, char_names=("Player", "Narrator", "Alice", "Bob")
    )
    client = MagicMock(spec=LLMClient)
    engine = Engine(client, db=mock_db)
    engine.start(scene)
    char = next(c for c in scene.character_pool if c.name == "Alice")
    # Add a visible conversation turn so the context is non-empty.
    ctx = engine.ctx
    assert ctx is not None
    ctx.user_message("Hello everyone.", name="Player")
    ctx.assistant_message("Hi Player.", tool_calls=[], name="Alice")
    state = MagicMock()
    state.engine = engine
    state.scene = scene
    state.ctx = ctx
    state.here_chars = engine.here_chars
    state.away_chars = engine.away_chars
    state.loc = scene.starting_location
    state.decision = None
    return engine, scene, char, state


def test_build_character_context_returns_system_prompt_and_messages() -> None:
    """Engine.build_character_context exposes the exact LLM view for a character."""
    engine, scene, char, _state = _make_engine_and_state()
    result = engine.build_character_context(char)

    assert "system_prompt" in result
    assert "messages" in result
    assert char.name in result["system_prompt"]
    assert any("Hi Player." in str(m.get("content", "")) for m in result["messages"])


def test_debug_context_command_returns_filtered_context() -> None:
    """The debug bridge 'context <name>' command dumps a character's LLM context."""
    _engine, scene, char, state = _make_engine_and_state()
    bridge = StructuredDebugBridge(state)
    result = bridge.run("context", [char.name])

    assert "error" not in result
    assert result["character"] == char.name
    assert "system_prompt" in result
    assert "messages" in result
    assert result["has_tools"] is True


def test_debug_context_command_rejects_unknown_character() -> None:
    """The context command reports an error for an unknown character name."""
    _engine, _scene, _char, state = _make_engine_and_state()
    bridge = StructuredDebugBridge(state)
    result = bridge.run("context", ["Zaphod"])

    assert "error" in result
    assert "Zaphod" in result["error"]
