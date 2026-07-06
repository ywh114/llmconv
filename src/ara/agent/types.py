"""Message types for the agent protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AgentRequest:
    """Incoming request from an agent client."""

    id: int
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, line: str) -> AgentRequest:
        """Parse a JSON line into a request."""
        data = json.loads(line)
        return cls(
            id=data["id"],
            method=data["method"],
            params=data.get("params", {}),
        )


@dataclass
class AgentResponse:
    """Outgoing response to an agent client."""

    id: int
    result: Any | None = None
    error: str | None = None

    def to_json(self) -> str:
        """Serialize the response to a JSON line."""
        return json.dumps(asdict(self), default=str) + "\n"


@dataclass
class StepResult:
    """Structured result of a single :meth:`Story.step` call."""

    event: str
    output: str = ""
    scene: dict[str, Any] | None = None
    suggestions: list[str] | None = None
    next_scene: str | None = None
