"""Entry point: python -m ara.webclient --port 8080"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from ara.agent.server import AgentServer
from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.webclient.proxy import DirectProxy
from ara.webclient.server import create_app
from ara.world.story import Story


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m ara.webclient``."""
    parser = argparse.ArgumentParser(
        prog="ara.webclient",
        description="Web VN gateway for Ara",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--agent-socket",
        default=None,
        help="Path to the Ara agent UNIX socket (default: data/sockets/ara_agent.sock)",
    )
    parser.add_argument(
        "--internal",
        action="store_true",
        help="Start an in-process agent server instead of connecting to a socket",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("data/assets/plot/demo/ini_scene.toml"),
        help="Initial scene TOML (only used with --internal)",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="LLM API key (overrides config and environment, only with --internal)",
    )
    parser.add_argument(
        "--client-step",
        type=int,
        default=0,
        help="Max queue depth: 0 = unlimited, 1 = old single-step bottleneck, N = pre-compute N events (only with --internal)",
    )
    args = parser.parse_args(argv)

    if args.internal:
        settings = AraSettings()
        if args.api_key:
            settings.api_key = args.api_key
        if not settings.api_key:
            settings.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        db = ChromaStore(settings)
        client = LLMClient(settings)
        story = Story(settings, db, client, args.scene)
        agent_server = AgentServer(story, socket_path=args.agent_socket, client_step=args.client_step)
        proxy = DirectProxy(agent_server)
        app = create_app(proxy=proxy)
        app.state.settings = settings
    else:
        app = create_app(socket_path=args.agent_socket)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
