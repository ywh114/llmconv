"""Agent API server for the Ara engine.

Uses a UNIX socket with newline-delimited JSON (NDJSON) for request/response.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

from ara.agent.debug_bridge import StructuredDebugBridge
from ara.agent.state import engine_to_dict, story_to_dict, scene_to_dict
from ara.agent.types import AgentRequest, AgentResponse
from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.utils.debug import _DebugState
from ara.world.story import Story


class ThreadingUnixStreamServer(
    socketserver.ThreadingMixIn, socketserver.UnixStreamServer
):
    """Thread-per-connection UNIX socket server."""

    daemon_threads = True
    allow_reuse_address = True


_ANSI_RE = re.compile(r"\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class AgentServer:
    """JSON-over-UNIX-socket server exposing the story engine to agents."""

    def __init__(
        self,
        story: Story,
        socket_path: str = "sockets/ara_agent.sock",
    ) -> None:
        self.story = story
        self.socket_path = socket_path
        self.lock = threading.Lock()
        self._server: ThreadingUnixStreamServer | None = None

    # ------------------------------------------------------------------ #
    # Request dispatch
    # ------------------------------------------------------------------ #

    def handle_request(self, request: AgentRequest) -> AgentResponse:
        try:
            result = self._dispatch(request.method, request.params)
            return AgentResponse(id=request.id, result=result)
        except Exception as exc:
            return AgentResponse(id=request.id, error=f"{type(exc).__name__}: {exc}")

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "start":
            scene_id = params.get("scene_id")
            with self.lock:
                self.story.start(scene_id=scene_id)
            return story_to_dict(self.story)

        if method == "step":
            with self.lock:
                try:
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        step = self.story.step()
                    output = _strip_ansi(buf.getvalue())
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    if "waiting for player input" in msg:
                        # Soft-return needs_player_input so the client
                        # doesn't treat an idle engine as a fatal error.
                        engine = self.story.engine
                        last_dec = engine.last_decision
                        suggestions = last_dec.suggestions if last_dec else []
                        speaker = (
                            engine.scene.player.name
                            if engine.scene and engine.scene.player
                            else None
                        )
                        return {
                            "event": "needs_player_input",
                            "output": "",
                            "scene": scene_to_dict(engine.scene)
                            if engine.scene
                            else None,
                            "suggestions": suggestions,
                            "next_scene": None,
                            "speaker": speaker,
                            "enter": [],
                            "exit": [],
                            "sprite_changes": {},
                        }
                    raise
            return {
                "event": step.event,
                "output": output,
                "scene": scene_to_dict(step.scene) if step.scene else None,
                "suggestions": step.suggestions,
                "next_scene": step.next_scene,
                "speaker": step.speaker,
                "enter": step.enter,
                "exit": step.exit,
                "sprite_changes": step.sprite_changes,
            }

        if method == "input":
            text = params.get("text") or ""
            with self.lock:
                self.story.submit_player_input(text)
            return {"submitted": text}

        if method == "generate":
            suggestion = params.get("suggestion", "")
            with self.lock:
                generated = self.story.generate_player_input(suggestion)
            return {"text": generated}

        if method == "state":
            with self.lock:
                return {
                    "story": story_to_dict(self.story),
                    "engine": engine_to_dict(self.story.engine),
                }

        if method == "run_until_input":
            outputs: list[str] = []
            events: list[dict[str, Any]] = []
            while True:
                with self.lock:
                    if self.story.finished:
                        break
                    if self.story.engine.needs_player_input:
                        # Emit a synthetic needs_player_input event so the
                        # client knows the engine is idle and waiting.
                        last_dec = self.story.engine.last_decision
                        suggestions = last_dec.suggestions if last_dec else []
                        speaker = (
                            self.story.engine.scene.player.name
                            if self.story.engine.scene
                            and self.story.engine.scene.player
                            else None
                        )
                        events.append({
                            "event": "needs_player_input",
                            "scene": scene_to_dict(self.story.engine.scene)
                            if self.story.engine.scene
                            else None,
                            "suggestions": suggestions,
                            "next_scene": None,
                            "speaker": speaker,
                            "enter": [],
                            "exit": [],
                        })
                        break
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        step = self.story.step()
                    output = _strip_ansi(buf.getvalue())
                outputs.append(output)
                events.append({
                    "event": step.event,
                    "output": output,
                    "scene": scene_to_dict(step.scene) if step.scene else None,
                    "suggestions": step.suggestions,
                    "next_scene": step.next_scene,
                    "speaker": step.speaker,
                    "enter": step.enter,
                    "exit": step.exit,
                    "sprite_changes": step.sprite_changes,
                })
                if step.event in (
                    "needs_player_input",
                    "scene_ended",
                    "story_complete",
                ):
                    break
            return {
                "events": events,
                "output": "".join(outputs),
            }

        if method == "reset":
            with self.lock:
                self.story.start()
            return story_to_dict(self.story)

        if method == "skip":
            scene_id = params.get("scene_id", "")
            with self.lock:
                step = self.story.jump_to(scene_id)
            return {
                "event": step.event,
                "output": "",
                "scene": scene_to_dict(step.scene) if step.scene else None,
                "suggestions": step.suggestions,
                "next_scene": step.next_scene,
                "speaker": step.speaker,
                "sprite_changes": step.sprite_changes,
            }

        if method == "debug":
            command = params.get("command", "")
            args = params.get("args", [])
            with self.lock:
                if self.story.current_scene is None:
                    return {"error": "No scene loaded."}
                state = _DebugState(
                    console=None,  # type: ignore[arg-type]
                    engine=self.story.engine,
                    scene=self.story.current_scene,
                    ctx=self.story.engine.ctx,
                    here_chars=self.story.engine.here_chars,
                    away_chars=self.story.engine.away_chars,
                    loc=self.story.engine.loc or self.story.current_scene.starting_location,
                    decision=self.story.engine.last_decision,
                )
                bridge = StructuredDebugBridge(state)
                return bridge.run(command, args)

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------ #
    # Server lifecycle
    # ------------------------------------------------------------------ #

    def start_listening(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        server_instance = self

        class _Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                for line in self.rfile:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        req = AgentRequest.from_json(line.decode("utf-8"))
                    except Exception as exc:
                        resp = AgentResponse(
                            id=-1, error=f"Parse error: {exc}"
                        )
                        self.wfile.write(resp.to_json().encode())
                        continue

                    resp = server_instance.handle_request(req)
                    self.wfile.write(resp.to_json().encode())
                    self.wfile.flush()

        self._server = ThreadingUnixStreamServer(self.socket_path, _Handler)
        try:
            self._server.serve_forever()
        finally:
            self._server.server_close()
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def shutdown(self) -> None:
        if self._server:
            self._server.shutdown()


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ara Agent API Server")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("data/assets/plot/0.toml"),
        help="Path to the initial scene TOML file",
    )
    parser.add_argument(
        "--socket",
        default="sockets/ara_agent.sock",
        help="UNIX socket path for the agent API",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="LLM API key (overrides config and environment)",
    )
    args = parser.parse_args(argv)

    settings = AraSettings()
    if args.api_key:
        settings.api_key = args.api_key
    if not settings.api_key:
        settings.api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    db = ChromaStore(settings)
    client = LLMClient(settings)
    story = Story(settings, db, client, args.scene)

    server = AgentServer(story, socket_path=args.socket)
    print(f"[agent-server] Listening on {args.socket}")
    server.start_listening()
    return 0


if __name__ == "__main__":
    sys.exit(main())
