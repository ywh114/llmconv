"""Thin NDJSON-over-UNIX-socket proxy to the Ara agent server."""

from __future__ import annotations

import json
import socket
from typing import Any


class AgentProxy:
    """Synchronous client that speaks the agent NDJSON protocol.

    Used by the web gateway to forward browser requests to the agent server.
    """

    def __init__(self, socket_path: str = "sockets/ara_agent.sock") -> None:
        self.socket_path = socket_path
        self._counter = 0

    def _call(self, method: str, **params: Any) -> Any:
        """Send a request and return the parsed JSON result."""
        self._counter += 1
        req = {"id": self._counter, "method": method, "params": params}
        payload = (json.dumps(req) + "\n").encode("utf-8")

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(30.0)
            sock.connect(self.socket_path)
            sock.sendall(payload)

            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Agent server closed connection")
                buf += chunk

        line, _ = buf.split(b"\n", 1)
        resp = json.loads(line.decode("utf-8"))
        if resp.get("error"):
            raise RuntimeError(resp["error"])
        return resp["result"]

    def start(self, scene_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if scene_id is not None:
            params["scene_id"] = scene_id
        return self._call("start", **params)

    def step(self) -> dict[str, Any]:
        return self._call("step")

    def input(self, text: str) -> dict[str, Any]:
        return self._call("input", text=text)

    def generate(self, suggestion: str) -> dict[str, Any]:
        return self._call("generate", suggestion=suggestion)

    def run_until_input(self) -> dict[str, Any]:
        return self._call("run_until_input")

    def state(self) -> dict[str, Any]:
        return self._call("state")

    def skip(self, scene_id: str) -> dict[str, Any]:
        return self._call("skip", scene_id=scene_id)

    def debug(self, command: str, args: list[str] | None = None) -> dict[str, Any]:
        if args is None:
            args = []
        return self._call("debug", command=command, args=args)
