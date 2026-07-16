"""Tests for :mod:`ara.llm.tools`."""

from __future__ import annotations


from ara.llm.tools import ToolRegistry, tool


class TestToolSchema:
    """Tests for the schema builder."""

    def test_basic_schema(self) -> None:
        """A basic tool schema should contain the required fields."""
        schema = tool(
            name="get_weather",
            description="Get the weather.",
            properties={"location": {"type": "string"}},
            required=["location"],
            strict=False,
        )
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "strict" not in schema["function"]

    def test_strict_schema(self) -> None:
        """Strict mode should inject ``strict`` and ``additionalProperties``."""
        schema = tool(
            name="get_weather",
            description="Get the weather.",
            properties={"location": {"type": "string"}},
            required=["location"],
            strict=True,
        )
        assert schema["function"]["strict"] is True
        assert schema["function"]["parameters"]["additionalProperties"] is False


class TestToolRegistry:
    """Tests for the tool hook registry."""

    def test_register_and_call(self) -> None:
        """Registered tools should be callable by name."""
        reg = ToolRegistry()
        reg.register("echo", lambda args: f"echo: {args}")
        result = reg.call("echo", "hello")
        assert result == "echo: hello"

    def test_missing_tool(self) -> None:
        """Calling an unregistered tool should return an error message."""
        reg = ToolRegistry()
        result = reg.call("missing", "{}")
        assert "not found" in result

    def test_contains(self) -> None:
        """``in`` should reflect registration status."""
        reg = ToolRegistry()
        assert "x" not in reg
        reg.register("x", lambda _: "")
        assert "x" in reg
