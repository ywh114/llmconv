"""Entry point: python -m ara.webclient --port 8080"""

from __future__ import annotations

import argparse

import uvicorn

from ara.webclient.server import create_app


def main(argv: list[str] | None = None) -> int:
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
        default="sockets/ara_agent.sock",
        help="Path to the Ara agent UNIX socket",
    )
    args = parser.parse_args(argv)

    app = create_app(socket_path=args.agent_socket)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
