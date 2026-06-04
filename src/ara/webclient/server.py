"""Starlette web gateway for the Ara VN frontend.

Simple JSON proxy to the agent server — no SSE, no streaming.
The frontend polls /next to advance the story and /input to reply.
"""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from ara.webclient.proxy import AgentProxy
from ara.utils.logger import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

_STATIC_DIR = Path(__file__).with_suffix("").parent / "static"
_ASSETS_DIR = (
    Path(__file__).with_suffix("").parent.parent.parent.parent / "data" / "assets"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalise_event(agent_result: dict[str, Any]) -> dict[str, Any]:
    """Convert an agent result into a clean frontend payload."""
    event_type = agent_result.get("event", "unknown")
    payload: dict[str, Any] = {"type": event_type}

    if event_type == "scene_loaded":
        scene = agent_result.get("scene") or {}
        payload["scene_id"] = scene.get("id")
        payload["location"] = scene.get("starting_location")
        payload["characters"] = scene.get("characters", [])
        payload["narrator"] = scene.get("narrator")
        payload["starting_characters"] = scene.get("starting_characters", [])
    elif event_type == "turn":
        payload["output"] = agent_result.get("output", "")
        payload["speaker"] = agent_result.get("speaker")
        payload["enter"] = agent_result.get("enter", [])
        payload["exit"] = agent_result.get("exit", [])
        payload["sprite_changes"] = agent_result.get("sprite_changes", {})
    elif event_type == "needs_player_input":
        payload["suggestions"] = agent_result.get("suggestions", [])
        payload["speaker"] = agent_result.get("speaker")
        payload["enter"] = agent_result.get("enter", [])
        payload["exit"] = agent_result.get("exit", [])
        payload["sprite_changes"] = agent_result.get("sprite_changes", {})
    elif event_type == "scene_ended":
        payload["next_scene"] = agent_result.get("next_scene")
    elif event_type == "story_complete":
        pass
    else:
        # Passthrough anything else (debug replies, etc.)
        payload = dict(agent_result)
        payload["type"] = event_type

    return payload


async def _proxy_call(proxy: AgentProxy, method: str, **kwargs: Any) -> Any:
    """Run a synchronous proxy call in a thread pool."""
    loop = asyncio.get_event_loop()
    fn = functools.partial(getattr(proxy, method), **kwargs)
    return await loop.run_in_executor(None, fn)


# --------------------------------------------------------------------------- #
# Route handlers
# --------------------------------------------------------------------------- #


async def _post_start(request: Request) -> JSONResponse:
    data = await request.json()
    scene_id = data.get("scene_id")
    proxy: AgentProxy = request.app.state.proxy
    try:
        result = await _proxy_call(proxy, "start", scene_id=scene_id)
        return JSONResponse(result)
    except Exception as exc:
        logger.warning(f"/start failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _post_next(request: Request) -> JSONResponse:
    """Advance until player input is needed, scene ends, or story completes."""
    proxy: AgentProxy = request.app.state.proxy
    try:
        result = await _proxy_call(proxy, "run_until_input")
        events = result.get("events", [])
        return JSONResponse(
            {
                "events": [_normalise_event(ev) for ev in events],
                "output": result.get("output", ""),
            }
        )
    except Exception as exc:
        logger.warning(f"/next failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _post_input(request: Request) -> JSONResponse:
    data = await request.json()
    text = data.get("text", "")
    proxy: AgentProxy = request.app.state.proxy
    try:
        result = await _proxy_call(proxy, "input", text=text)
        return JSONResponse(result)
    except Exception as exc:
        logger.warning(f"/input failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _post_generate(request: Request) -> JSONResponse:
    data = await request.json()
    suggestion = data.get("suggestion", "")
    proxy: AgentProxy = request.app.state.proxy
    try:
        result = await _proxy_call(proxy, "generate", suggestion=suggestion)
        return JSONResponse(result)
    except Exception as exc:
        logger.warning(f"/generate failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _post_step(request: Request) -> JSONResponse:
    proxy: AgentProxy = request.app.state.proxy
    try:
        result = await _proxy_call(proxy, "step")
        return JSONResponse(_normalise_event(result))
    except Exception as exc:
        logger.warning(f"/step failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _post_skip(request: Request) -> JSONResponse:
    data = await request.json()
    scene_id = data.get("scene_id", "")
    proxy: AgentProxy = request.app.state.proxy
    try:
        result = await _proxy_call(proxy, "skip", scene_id=scene_id)
        return JSONResponse(_normalise_event(result))
    except Exception as exc:
        logger.warning(f"/skip failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _post_debug(request: Request) -> JSONResponse:
    data = await request.json()
    command = data.get("command", "")
    args = data.get("args", [])
    proxy: AgentProxy = request.app.state.proxy
    try:
        result = await _proxy_call(proxy, "debug", command=command, args=args)
        return JSONResponse(result)
    except Exception as exc:
        logger.warning(f"/debug failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _get_state(request: Request) -> JSONResponse:
    proxy: AgentProxy = request.app.state.proxy
    try:
        result = await _proxy_call(proxy, "state")
        return JSONResponse(result)
    except Exception as exc:
        logger.warning(f"/state failed: {exc}")
        return JSONResponse({"error": str(exc)}, status_code=500)


async def _index(request: Request) -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #


def create_app(socket_path: str = "sockets/ara_agent.sock") -> Starlette:
    """Create and configure the Starlette gateway application."""
    app = Starlette(
        debug=True,
        routes=[
            Route("/", _index),
            Route("/start", _post_start, methods=["POST"]),
            Route("/next", _post_next, methods=["POST"]),
            Route("/input", _post_input, methods=["POST"]),
            Route("/generate", _post_generate, methods=["POST"]),
            Route("/step", _post_step, methods=["POST"]),
            Route("/skip", _post_skip, methods=["POST"]),
            Route("/debug", _post_debug, methods=["POST"]),
            Route("/state", _get_state),
            Mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static"),
            Mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets"),
        ],
    )
    app.state.proxy = AgentProxy(socket_path=socket_path)
    return app
