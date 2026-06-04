"""Synchronous client for the Ara agent API."""

from __future__ import annotations

import json
import socket
from typing import Any


class AgentClient:
    """JSON-over-UNIX-socket client for the Ara agent server."""

    def __init__(self, socket_path: str = "sockets/ara_agent.sock") -> None:
        self.socket_path = socket_path
        self._sock: socket.socket | None = None
        self._counter = 0

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.socket_path)

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> AgentClient:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Low-level protocol
    # ------------------------------------------------------------------ #

    def _call(self, method: str, **params: Any) -> Any:
        if self._sock is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        self._counter += 1
        req = {"id": self._counter, "method": method, "params": params}
        self._sock.sendall((json.dumps(req) + "\n").encode("utf-8"))

        # Read exactly one line (one response)
        buf = b""
        while b"\n" not in buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Server closed connection")
            buf += chunk
        line, _ = buf.split(b"\n", 1)
        resp = json.loads(line.decode("utf-8"))
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["result"]

    # ------------------------------------------------------------------ #
    # High-level API
    # ------------------------------------------------------------------ #

    def start(self, scene_id: str | None = None) -> dict[str, Any]:
        """Start or restart the story.

        :param scene_id: Optional scene identifier to jump to immediately.
        """
        params: dict[str, Any] = {}
        if scene_id is not None:
            params["scene_id"] = scene_id
        return self._call("start", **params)

    def step(self) -> dict[str, Any]:
        """Advance the story by one tick.

        Returns a dict with keys: ``event``, ``output``, ``scene``,
        ``suggestions``, ``next_scene``.
        """
        return self._call("step")

    def input(self, text: str) -> dict[str, Any]:
        """Submit player input."""
        return self._call("input", text=text)

    def state(self) -> dict[str, Any]:
        """Get a full state snapshot (story + engine)."""
        return self._call("state")

    def run_until_input(self) -> dict[str, Any]:
        """Auto-step until player input is required or the story ends.

        Returns a dict with keys: ``events``, ``output``.
        """
        return self._call("run_until_input")

    def reset(self) -> dict[str, Any]:
        """Reset the story to the beginning."""
        return self._call("reset")

    def skip(self, scene_id: str) -> dict[str, Any]:
        """Jump to a specific scene, abandoning the current one."""
        return self._call("skip", scene_id=scene_id)

    def debug(self, command: str, args: list[str] | None = None) -> dict[str, Any]:
        """Execute a debug command and return structured output.

        :param command: Debug command name (dump, info, here, away,
            loc, scene, decision, scratch, exec, help).
        :param args: Optional list of positional arguments for the command.
        """
        if args is None:
            args = []
        return self._call("debug", command=command, args=args)
