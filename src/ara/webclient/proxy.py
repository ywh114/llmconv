"""Proxy to the Ara agent server.

Supports both UNIX-socket mode (standalone agent server) and direct mode
(internal agent server running in the same process).
"""

from __future__ import annotations

import json
import socket
from typing import Any

from ara.agent.server import AgentServer

# Socket timeout (seconds) for methods without an explicit entry in
# ``_METHOD_TIMEOUTS``.
_DEFAULT_TIMEOUT = 30.0

# Per-method socket timeouts.  Methods that can trigger LLM generation get a
# much longer budget than the default.
_METHOD_TIMEOUTS: dict[str, float] = {
    "start": 300.0,
    "step": 300.0,
    "input": 300.0,
    "generate": 300.0,
    "run_until_input": 300.0,
    "load": 300.0,
}


class BaseProxy:
    """Shared agent-API method implementations.

    Subclasses implement :meth:`_call` to transport a single
    ``(method, params)`` request and return the decoded result, raising on
    error.
    """

    def _call(self, method: str, **params: Any) -> Any:
        raise NotImplementedError

    def start(self, scene_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if scene_id is not None:
            params["scene_id"] = scene_id
        return self._call("start", **params)

    def step(self) -> dict[str, Any]:
        return self._call("step")

    def input(
        self,
        text: str,
        attempt: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"text": text}
        if attempt is not None:
            params["attempt"] = attempt
        return self._call("input", **params)

    def attempt(self, text: str) -> dict[str, Any]:
        return self._call("attempt", text=text)

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

    def save(self, slot: int = 1) -> dict[str, Any]:
        return self._call("save", slot=slot)

    def load(self, slot: int = 1) -> dict[str, Any]:
        return self._call("load", slot=slot)

    def list_saves(self, story_id: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if story_id is not None:
            params["story_id"] = story_id
        return self._call("list_saves", **params)

    def delete_save(self, slot: int, story_id: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"slot": slot}
        if story_id is not None:
            params["story_id"] = story_id
        return self._call("delete_save", **params)


class AgentProxy(BaseProxy):
    """Synchronous client that speaks the agent NDJSON protocol.

    Used by the web gateway to forward browser requests to the agent server.
    """

    def __init__(self, socket_path: str | None = None) -> None:
        from ara.config import AraSettings
        self.socket_path = socket_path or str(AraSettings().default_socket_path)
        self._counter = 0

    def _call(self, method: str, **params: Any) -> Any:
        """Send a request and return the parsed JSON result."""
        self._counter += 1
        req = {"id": self._counter, "method": method, "params": params}
        payload = (json.dumps(req) + "\n").encode("utf-8")

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_METHOD_TIMEOUTS.get(method, _DEFAULT_TIMEOUT))
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


class DirectProxy(BaseProxy):
    """Direct in-process proxy to an AgentServer instance.

    Avoids UNIX socket overhead by calling :meth:`AgentServer.handle_request`
    directly.  Thread-safe via the server's internal lock.
    """

    def __init__(self, server: AgentServer) -> None:
        self.server = server
        self._counter = 0

    def _call(self, method: str, **params: Any) -> Any:
        from ara.agent.types import AgentRequest
        self._counter += 1
        req = AgentRequest(id=self._counter, method=method, params=params)
        resp = self.server.handle_request(req)
        if resp.error:
            raise RuntimeError(resp.error)
        return resp.result
