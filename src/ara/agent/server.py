"""Agent API server for the Ara engine.

Uses a UNIX socket with newline-delimited JSON (NDJSON) for request/response.
"""

from __future__ import annotations

import argparse
import collections
import os
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any

from ara.agent.debug_bridge import StructuredDebugBridge
from ara.agent.state import build_visual_state, engine_to_dict, story_to_dict, scene_to_dict, location_to_dict
from ara.agent.types import AgentRequest, AgentResponse
from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.persistence.save import SaveManager
from ara.utils.debug import _DebugState
from ara.utils.logger import get_logger
from ara.world.story import Story

logger = get_logger(__name__)


class ThreadingUnixStreamServer(
    socketserver.ThreadingMixIn, socketserver.UnixStreamServer
):
    """Thread-per-connection UNIX socket server."""

    daemon_threads = True
    allow_reuse_address = True


def _step_to_dict(step: Any) -> dict[str, Any]:
    """Convert a StoryStep to a JSON-friendly dict."""
    return {
        "event": step.event,
        "phase": step.phase,
        "output": step.output,
        "inner": step.inner,
        "scene": scene_to_dict(step.scene) if step.scene else None,
        "suggestions": step.suggestions,
        "next_scene": step.next_scene,
        "next_scene_name": step.next_scene_name,
        "loading_background": step.loading_background,
        "speaker": step.speaker,
        "speaker_title": step.speaker_title,
        "enter": step.enter,
        "exit": step.exit,
        "spawn": step.spawn,
        "sprite_changes": step.sprite_changes,
        "switch_background": step.switch_background,
        "system_changes": step.system_changes,
        "location": location_to_dict(step.location) if step.location else None,
    }


class AgentServer:
    """JSON-over-UNIX-socket server exposing the story engine to agents.

    Story events are precomputed by a background worker and buffered in a
    queue.  ``/step`` pops from the queue, decoupling the client from LLM
    latency.
    """

    def __init__(
        self,
        story: Story,
        socket_path: str | None = None,
        client_step: int = 0,
    ) -> None:
        """Create a server for *story*.

        :param story: The story instance to expose.
        :param socket_path: Path for the UNIX socket.  Defaults to
            :attr:`AraSettings.default_socket_path`.
        :param client_step: Maximum number of events to precompute before
            pausing the worker (``0`` means unlimited).
        """
        self.story = story
        self.socket_path = str(socket_path) if socket_path else str(AraSettings().default_socket_path)
        self._server: ThreadingUnixStreamServer | None = None
        self.client_step = client_step

        # Queue state
        self._event_queue: collections.deque[dict[str, Any]] = collections.deque()
        self._queue_lock = threading.Lock()
        self._queue_condition = threading.Condition(self._queue_lock)

        # Story state
        self._story_lock = threading.Lock()

        # Cached snapshots for non-blocking reads
        self._last_state: dict[str, Any] | None = None
        self._last_save_snapshot: dict[str, Any] | None = None
        self._last_visual_state: dict[str, Any] | None = None

        # Worker lifecycle
        self._worker_alive = False
        self._worker_running = False
        self._worker_thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # Worker
    # ------------------------------------------------------------------ #

    def _worker_loop(self) -> None:
        """Background worker: advance the story and enqueue events."""
        while True:
            with self._queue_lock:
                if not self._worker_alive:
                    return
                if not self._worker_running:
                    self._queue_condition.wait()
                    continue
                # Double-check after waking
                if not self._worker_alive:
                    return

            # Step the story (holds story lock for the duration of the call)
            try:
                with self._story_lock:
                    if not self._worker_alive:
                        return
                    step = self.story.step()

                    # Build read-only snapshots while state is consistent
                    state_snapshot = {
                        "story": story_to_dict(self.story),
                        "engine": engine_to_dict(self.story.engine),
                    }
                    save_snapshot = SaveManager(self.story.config)._build_snapshot(
                        self.story
                    )
                    visual_snapshot = build_visual_state(self.story)
            except RuntimeError:
                # Engine errors (e.g. story complete, waiting for input)
                # Stop the worker and let the client deal with it.
                with self._queue_lock:
                    self._worker_running = False
                    self._queue_condition.notify_all()
                continue
            except Exception:
                logger.exception('Background worker encountered an unexpected error')
                with self._queue_lock:
                    self._worker_running = False
                    self._queue_condition.notify_all()
                continue

            event = _step_to_dict(step)

            with self._queue_lock:
                self._event_queue.append(event)
                self._queue_condition.notify_all()

                # Capture queue into snapshots for consistent non-blocking reads
                state_snapshot["queue"] = list(self._event_queue)
                save_snapshot["queue"] = list(self._event_queue)
                self._last_state = state_snapshot
                self._last_save_snapshot = save_snapshot
                self._last_visual_state = visual_snapshot

                # Stop conditions
                if step.event == "needs_player_input":
                    self._worker_running = False
                elif step.event == "story_complete":
                    self._worker_running = False
                elif self.client_step > 0 and len(self._event_queue) >= self.client_step:
                    self._worker_running = False

                self._queue_condition.notify_all()

    def _spawn_worker(self) -> None:
        """Start a fresh worker thread."""
        logger.info("Spawning background worker")
        # Install a fresh cancel event for this worker.  _kill_worker()
        # will set it when the user asks for load/reset/start.
        client = getattr(self.story, 'client', None)
        if client is not None:
            client.cancel_event = threading.Event()
        with self._queue_lock:
            self._worker_alive = True
            self._worker_running = True
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
            self._queue_condition.notify_all()

    def _kill_worker(self) -> None:
        """Signal the worker to stop and wait for it to die."""
        logger.warning("Killing background worker")
        with self._queue_lock:
            self._worker_alive = False
            self._worker_running = False
            self._queue_condition.notify_all()

        # Cancel any in-flight LLM call so the worker releases _story_lock
        # quickly instead of blocking /load, /state, or /save.
        client = getattr(self.story, 'client', None)
        if client is not None:
            if getattr(client, 'cancel_event', None) is None:
                client.cancel_event = threading.Event()
            client.cancel_event.set()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=30.0)

    # ------------------------------------------------------------------ #
    # Request dispatch
    # ------------------------------------------------------------------ #

    def handle_request(self, request: AgentRequest) -> AgentResponse:
        """Dispatch a single request and return a response.

        Exceptions are converted into error responses so the connection stays
        open.
        """
        try:
            result = self._dispatch(request.method, request.params)
            return AgentResponse(id=request.id, result=result)
        except Exception as exc:
            logger.exception(f'Error handling request {request.method}')
            return AgentResponse(id=request.id, error=f"{type(exc).__name__}: {exc}")

    def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method == "start":
            scene_id = params.get("scene_id")
            self._kill_worker()
            with self._story_lock:
                with self._queue_lock:
                    self._event_queue.clear()
                # Invalidate snapshots so reads never see the previous run.
                self._last_state = None
                self._last_save_snapshot = None
                self._last_visual_state = None
                # Always clear persisted character/story memory on a fresh start
                # so previous runs cannot pollute the new session.
                self.story.start(scene_id=scene_id, clear_history=True)
            self._spawn_worker()
            return story_to_dict(self.story)

        if method == "step":
            with self._queue_lock:
                while not self._event_queue:
                    with self._story_lock:
                        needs_input = self.story.engine.needs_player_input

                    if needs_input:
                        # Engine is waiting for input but the queue is empty
                        # (client already consumed the needs_player_input event).
                        # Return a synthetic event so the client UX is consistent.
                        engine = self.story.engine
                        last_dec = engine.last_decision
                        suggestions = last_dec.suggestions if last_dec else []
                        speaker = (
                            engine.scene.player.name
                            if engine.scene and engine.scene.player
                            else None
                        )
                        logger.info(
                            f"[STEP_DEBUG] synthetic needs_player_input, "
                            f"last_decision_next={last_dec.next_char.name if last_dec else None}, "
                            f"suggestions={suggestions}"
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
                            "location": location_to_dict(engine.loc)
                            if engine.loc
                            else None,
                        }

                    # Worker crashed or paused while engine is ready — restart it.
                    if not self._worker_running and self._worker_alive:
                        with self._story_lock:
                            if not self.story.engine.needs_player_input and not self.story.finished:
                                logger.warning("Worker appears dead; restarting.")
                                self._worker_running = True
                                self._queue_condition.notify_all()

                    self._queue_condition.wait(timeout=0.5)

                if not self._event_queue:
                    with self._story_lock:
                        if self.story.engine.needs_player_input:
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
                                "location": location_to_dict(engine.loc)
                                if engine.loc
                                else None,
                            }
                    raise RuntimeError("No events available")

                event = self._event_queue[0]
                logger.info(
                    f"[STEP_DEBUG] popping event={event.get('event')}, "
                    f"speaker={event.get('speaker')}, queue_len={len(self._event_queue)}, "
                    f"engine_needs_input={self.story.engine.needs_player_input}"
                )

                # story_complete is the terminal event. Once the story is
                # finished, leave the event in the queue so every subsequent
                # /step returns it instead of raising.
                if event["event"] != "story_complete":
                    self._event_queue.popleft()

                # Wake worker if it was sleeping due to queue capacity
                if (
                    self.client_step > 0
                    and len(self._event_queue) < self.client_step
                    and not self._worker_running
                    and self._worker_alive
                ):
                    with self._story_lock:
                        needs_input = self.story.engine.needs_player_input
                        finished = self.story.finished
                    if not needs_input and not finished:
                        self._worker_running = True
                        self._queue_condition.notify()

                return event

        if method == "attempt":
            text = params.get("text") or ""
            with self._story_lock:
                self.story.submit_attempt(text)
            return {"attempted": text}

        if method == "input":
            text = params.get("text") or ""
            attempt = params.get("attempt")
            with self._story_lock:
                logger.info(
                    f"[INPUT_DEBUG] needs_player_input={self.story.engine.needs_player_input}, "
                    f"queue_len={len(self._event_queue)}, "
                    f"front_event={self._event_queue[0].get('event') if self._event_queue else None}, "
                    f"last_decision_next={self.story.engine.last_decision.next_char.name if self.story.engine.last_decision else None}"
                )
                self.story.submit_player_input(text, attempt=attempt)
                # Keep the visual snapshot fresh so a client reattaching
                # right after submitting still sees this line in its backlog.
                self._last_visual_state = build_visual_state(self.story)
                with self._queue_lock:
                    # If the client called /input before popping the
                    # needs_player_input event, drain it now.
                    if (
                        self._event_queue
                        and self._event_queue[0].get("event") == "needs_player_input"
                    ):
                        self._event_queue.popleft()
                    self._worker_running = True
                    self._queue_condition.notify()
            return {"submitted": text}

        if method == "generate":
            suggestion = params.get("suggestion", "")
            with self._story_lock:
                generated = self.story.generate_player_input(suggestion)
            return {"text": generated}

        if method == "state":
            acquired = self._story_lock.acquire(blocking=False)
            if acquired:
                try:
                    state = {
                        "story": story_to_dict(self.story),
                        "engine": engine_to_dict(self.story.engine),
                    }
                    with self._queue_lock:
                        state["queue"] = list(self._event_queue)
                    self._last_state = state
                    return state
                finally:
                    self._story_lock.release()

            # Lock held by worker - serve cached snapshot
            with self._queue_lock:
                if self._last_state is not None:
                    return dict(self._last_state)

            # No cache yet - fall back to blocking
            with self._story_lock:
                state = {
                    "story": story_to_dict(self.story),
                    "engine": engine_to_dict(self.story.engine),
                }
                with self._queue_lock:
                    state["queue"] = list(self._event_queue)
                self._last_state = state
                return state

        if method == "run_until_input":
            # Deprecated: drain the queue until a stopping event.
            events: list[dict[str, Any]] = []
            outputs: list[str] = []
            while True:
                with self._queue_lock:
                    if not self._event_queue:
                        with self._story_lock:
                            finished = self.story.finished
                            needs_input = self.story.engine.needs_player_input
                        if finished:
                            break
                        if needs_input:
                            engine = self.story.engine
                            last_dec = engine.last_decision
                            suggestions = last_dec.suggestions if last_dec else []
                            speaker = (
                                engine.scene.player.name
                                if engine.scene and engine.scene.player
                                else None
                            )
                            events.append({
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
                            })
                            break
                        self._queue_condition.wait(timeout=0.5)
                        continue

                    event = self._event_queue.popleft()
                    events.append(event)
                    outputs.append(event.get("output", ""))

                    # Wake worker if capacity freed up
                    if (
                        self.client_step > 0
                        and len(self._event_queue) < self.client_step
                        and not self._worker_running
                        and self._worker_alive
                    ):
                        with self._story_lock:
                            needs_input = self.story.engine.needs_player_input
                            finished = self.story.finished
                        if not needs_input and not finished:
                            self._worker_running = True
                            self._queue_condition.notify()

                    if event["event"] in (
                        "needs_player_input",
                        "transition",
                        "story_complete",
                    ):
                        break

            return {
                "events": events,
                "output": "".join(outputs),
            }

        if method == "save":
            slot = int(params.get("slot", 1))
            acquired = self._story_lock.acquire(blocking=False)
            if acquired:
                try:
                    with self._queue_lock:
                        queue = list(self._event_queue)
                    manager = SaveManager(self.story.config)
                    path = manager.save(self.story, slot, queue=queue)
                    self._last_save_snapshot = manager._build_snapshot(
                        self.story, queue=queue
                    )
                    return {"slot": slot, "path": str(path)}
                finally:
                    self._story_lock.release()

            # Lock held by worker - write from cached snapshot
            with self._queue_lock:
                snapshot = self._last_save_snapshot
            if snapshot is not None:
                manager = SaveManager(self.story.config)
                path = manager.save_snapshot(dict(snapshot), slot)
                return {"slot": slot, "path": str(path)}

            # No cache yet - fall back to blocking
            with self._story_lock:
                with self._queue_lock:
                    queue = list(self._event_queue)
                manager = SaveManager(self.story.config)
                path = manager.save(self.story, slot, queue=queue)
                return {"slot": slot, "path": str(path)}

        if method == "load":
            slot = int(params.get("slot", 1))
            self._kill_worker()
            with self._story_lock:
                manager = SaveManager(self.story.config)
                queue = manager.load(self.story, slot)
                with self._queue_lock:
                    self._event_queue.clear()
                    self._event_queue.extend(queue)
                self._last_state = None
                self._last_save_snapshot = None
                self._last_visual_state = build_visual_state(self.story)
            self._spawn_worker()
            return dict(self._last_visual_state)

        if method == "continue":
            # Reattach a client (e.g. after a browser reload) without touching
            # engine state: serve the worker-published snapshot so this never
            # waits on an in-flight LLM turn.
            with self._queue_lock:
                snapshot = self._last_visual_state
            if snapshot is None:
                # No event processed yet (right after /start): compute live.
                with self._story_lock:
                    if self.story.current_scene is None or self.story.finished:
                        return {"active": False}
                    snapshot = build_visual_state(self.story)
            if snapshot.get("scene") is None or snapshot.get("finished", False):
                return {"active": False}
            result = dict(snapshot)  # copy; don't mutate the cache
            result["active"] = True
            return result

        if method == "list_saves":
            story_id = params.get("story_id") or self.story._story_dir.name
            manager = SaveManager(self.story.config)
            saves = manager.list_saves(story_id)
            return [
                {
                    "slot": s.slot,
                    "story_id": s.story_id,
                    "scene_id": s.scene_id,
                    "timestamp": s.timestamp,
                    "scene_history": s.scene_history,
                }
                for s in saves
            ]

        if method == "delete_save":
            slot = int(params.get("slot", 1))
            story_id = params.get("story_id") or self.story._story_dir.name
            manager = SaveManager(self.story.config)
            manager.delete(story_id, slot)
            return {"deleted": slot}

        if method == "reset":
            self._kill_worker()
            with self._story_lock:
                with self._queue_lock:
                    self._event_queue.clear()
                self.story.start(clear_history=True)
                self._last_state = None
                self._last_save_snapshot = None
                self._last_visual_state = build_visual_state(self.story)
            self._spawn_worker()
            return dict(self._last_visual_state)

        if method == "skip":
            scene_id = params.get("scene_id", "")
            self._kill_worker()
            with self._story_lock:
                with self._queue_lock:
                    self._event_queue.clear()
                self._last_state = None
                self._last_save_snapshot = None
                self._last_visual_state = None
                step = self.story.jump_to(scene_id)
            self._spawn_worker()
            return _step_to_dict(step)

        if method == "debug":
            command = params.get("command", "")
            args = params.get("args", [])
            with self._story_lock:
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
        """Start the UNIX socket server and serve requests forever."""
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
        """Stop the background worker and shut down the socket server."""
        self._kill_worker()
        if self._server:
            self._server.shutdown()


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m ara.agent``."""
    parser = argparse.ArgumentParser(description="Ara Agent API Server")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("data/assets/plot/0.toml"),
        help="Path to the initial scene TOML file",
    )
    parser.add_argument(
        "--socket",
        default=str(AraSettings().default_socket_path),
        help="UNIX socket path for the agent API",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="LLM API key (overrides config and environment)",
    )
    parser.add_argument(
        "--client-step",
        type=int,
        default=0,
        help="Max queue depth (0 = infinite). Preserves client-side pacing.",
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

    server = AgentServer(story, socket_path=args.socket, client_step=args.client_step)
    print(f"[agent-server] Listening on {args.socket}")
    server.start_listening()
    return 0


if __name__ == "__main__":
    sys.exit(main())
