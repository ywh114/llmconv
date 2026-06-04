"""Tool schema builder and lightweight hook registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


def tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    strict: bool = True,
) -> dict[str, Any]:
    """Build an OpenAI-compatible function-tool schema dict.

    When *strict* is ``True`` the dict includes ``strict: true`` and
    ``additionalProperties: false`` so that DeepSeek's beta endpoint can
    enforce schema compliance.

    :param name: Function name exposed to the LLM.
    :param description: Human-readable description of what the tool does.
    :param properties: JSON-schema ``properties`` mapping for the arguments.
    :param required: List of argument names that must be supplied.
    :param strict: Whether to enable strict JSON-schema mode.
    :return: A dict suitable for the ``tools`` parameter of the chat-completion
        API.
    """
    schema: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    if strict:
        schema["function"]["strict"] = True
        schema["function"]["parameters"]["additionalProperties"] = False
    return schema


@dataclass
class ToolRegistry:
    """Simple name-to-callable mapping for tool execution.

    The orchestrator and character agents register their tool implementations
    here.  After the LLM returns tool calls, the registry dispatches each
    call to its registered Python function.
    """

    _hooks: dict[str, Callable[..., str]] = field(default_factory=dict)
    """Internal mapping from tool name to implementation."""

    def register(self, name: str, fn: Callable[..., str]) -> None:
        """Register *fn* as the handler for tool *name*.

        :param name: Tool name exactly as declared in the schema.
        :param fn: Callable that receives the JSON-encoded arguments string
            and returns a string result.
        """
        self._hooks[name] = fn

    def call(self, name: str, args: str) -> str:
        """Invoke the registered handler for *name*.

        :param name: Tool name.
        :param args: JSON-encoded arguments string.
        :return: The handler's return value, or an error message if the tool
            is not registered.
        """
        fn = self._hooks.get(name)
        if fn is None:
            return f"Error: tool '{name}' not found."
        return fn(args)

    def __contains__(self, name: str) -> bool:
        """Return ``True`` if a handler is registered for *name*."""
        return name in self._hooks
