"""First-class CLI client for the Ara Agent API.

Usage
-----
Start / step / reply (server auto-starts on first use)::

    $ python examples/aractl.py start
    $ python examples/aractl.py step
    $ python examples/aractl.py next
    $ python examples/aractl.py reply "Hello there"
    $ python examples/aractl.py reply "/info"
    $ python examples/aractl.py debug dump
    $ python examples/aractl.py state

Force a fresh server after code changes::

    $ python examples/aractl.py --restart step

Shut down the daemon::

    $ python examples/aractl.py --kill

Output is pretty-printed by default; pass ``--json`` for machine-readable JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any

# Ensure project src is on path when run directly
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from ara.agent.client import AgentClient

DEFAULT_SOCKET = "sockets/ara_agent.sock"
PID_FILE = pathlib.Path("sockets/ara_agent_cli.pid")


# ------------------------------------------------------------------ #
# Server lifecycle
# ------------------------------------------------------------------ #


def _server_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        return None


def _is_server_running(socket_path: str) -> bool:
    import socket

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(0.3)
        sock.connect(socket_path)
        sock.close()
        return True
    except OSError:
        return False


def _kill_server(socket_path: str) -> bool:
    killed = False
    pid = _server_pid()
    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
            killed = True
        except ProcessLookupError:
            pass
        PID_FILE.unlink(missing_ok=True)
    # Also unlink stale socket
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass
    return killed


def _start_server_daemon(socket_path: str, scene_path: pathlib.Path, api_key: str = "") -> None:
    """Start the AgentServer as a background daemon process."""
    cmd = [
        sys.executable,
        "-m",
        "ara.agent.server",
        "--socket",
        socket_path,
        "--scene",
        str(scene_path),
    ]
    if api_key:
        cmd += ["--api-key", api_key]

    # Clean up stale socket
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))

    # Wait for server to be ready (Chroma init + scene load can take a while)
    for _ in range(200):
        if _is_server_running(socket_path):
            return
        time.sleep(0.05)
    raise RuntimeError("Server failed to start within 10 seconds")


# ------------------------------------------------------------------ #
# Client helpers
# ------------------------------------------------------------------ #


def _get_client(args: argparse.Namespace) -> AgentClient:
    if not _is_server_running(args.socket):
        if args.restart:
            _kill_server(args.socket)
        _start_server_daemon(
            args.socket,
            pathlib.Path(args.scene),
            api_key=args.api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
        )
    client = AgentClient(args.socket)
    client.connect()
    return client


# ------------------------------------------------------------------ #
# Formatting
# ------------------------------------------------------------------ #


def _fmt_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


def _fmt_step(data: dict[str, Any]) -> str:
    lines: list[str] = []
    event = data.get("event", "unknown")
    lines.append(f"[{event.upper()}]")

    scene = data.get("scene")
    if scene is not None:
        lines.append(f"  scene: {scene.get('id')}")
        loc = scene.get("starting_location")
        if loc:
            lines.append(f"  location: {loc}")
        chars = scene.get("characters")
        if chars:
            lines.append(f"  characters: {', '.join(c['name'] for c in chars)}")

    suggestions = data.get("suggestions")
    if suggestions:
        lines.append("  suggestions:")
        for s in suggestions:
            lines.append(f"    - {s}")

    next_scene = data.get("next_scene")
    if next_scene is not None:
        lines.append(f"  next_scene: {next_scene}")

    output = data.get("output", "")
    if output.strip():
        lines.append("")
        lines.append(output.rstrip())

    return "\n".join(lines)


def _fmt_run(data: dict[str, Any]) -> str:
    lines: list[str] = []
    events = data.get("events", [])
    for ev in events:
        lines.append(_fmt_step(ev))
        lines.append("")
    output = data.get("output", "")
    if output.strip():
        lines.append(output.rstrip())
    return "\n".join(lines).rstrip()


def _fmt_state(data: dict[str, Any]) -> str:
    lines: list[str] = ["[STATE]"]
    story = data.get("story", {})
    engine = data.get("engine", {})

    lines.append(f"  finished: {story.get('finished')}")
    lines.append(f"  scene_history: {story.get('scene_history', [])}")
    cur = story.get("current_scene")
    if cur:
        lines.append(f"  current_scene: {cur.get('id')}")
    else:
        lines.append("  current_scene: (none)")

    lines.append(f"  location: {engine.get('location', 'N/A')}")
    lines.append(f"  here: {engine.get('here', [])}")
    lines.append(f"  away: {engine.get('away', [])}")
    lines.append(f"  needs_player_input: {engine.get('needs_player_input')}")
    lines.append(f"  context_length: {engine.get('context_length')}")
    dec = engine.get("last_decision")
    if dec:
        lines.append(f"  last_decision: {dec.get('next_char')} — {dec.get('directive') or '(no directive)'}")
    else:
        lines.append("  last_decision: (none)")

    return "\n".join(lines)


def _fmt_debug(data: dict[str, Any]) -> str:
    lines: list[str] = ["[DEBUG]"]
    if "error" in data:
        lines.append(f"  error: {data['error']}")
        return "\n".join(lines)
    # Generic pretty-print of all keys
    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f"  {key}:")
            for item in value:
                if isinstance(item, dict):
                    line = ", ".join(f"{k}={v}" for k, v in item.items())
                    lines.append(f"    - {line}")
                else:
                    lines.append(f"    - {item}")
        elif isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                lines.append(f"    {k}: {v}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _print(data: Any, args: argparse.Namespace) -> None:
    if args.json:
        print(_fmt_json(data))
    elif isinstance(data, dict):
        # Try to dispatch to a prettier formatter
        if "event" in data:
            print(_fmt_step(data))
        elif "events" in data:
            print(_fmt_run(data))
        elif "story" in data and "engine" in data:
            print(_fmt_state(data))
        elif "submitted" in data:
            print(f"[INPUT] submitted: {data['submitted']!r}")
        else:
            print(_fmt_debug(data))
    else:
        print(data)


# ------------------------------------------------------------------ #
# Subcommands
# ------------------------------------------------------------------ #


def cmd_start(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.start(scene_id=args.scene_id)
    _print(result, args)
    return 0


def cmd_step(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.step()
    _print(result, args)
    return 0


def cmd_next(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.run_until_input()
    _print(result, args)
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.state()
    _print(result, args)
    return 0


def cmd_reply(args: argparse.Namespace) -> int:
    text = " ".join(args.text)
    client = _get_client(args)
    stripped = text.strip()
    if stripped.startswith("/"):
        # First-class debug via reply prefix
        parts = stripped[1:].split()
        if not parts:
            print("[ERROR] empty debug command")
            return 1
        result = client.debug(parts[0], args=parts[1:])
    else:
        result = client.input(text)
    _print(result, args)
    return 0


def cmd_skip(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.skip(args.scene_id)
    _print(result, args)
    return 0


def cmd_debug(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.debug(args.command, args=args.args or [])
    _print(result, args)
    return 0


def cmd_help(args: argparse.Namespace) -> int:
    print(
        """Ara Agent CLI — first-class client for the game server.

Subcommands:
  start                Start or restart the story.
  step                 Advance one tick.
  next                 Auto-step until player input is required.
  skip <scene_id>      Jump to a specific scene (abandons current scene).
  state                Show full state snapshot.
  reply <text>         Submit player input. If text starts with /,
                       it is treated as a debug command (first-class).
  debug <cmd> [args..] Run an explicit debug command.
  help                 Show this message.

Global options:
  --json               Output raw JSON (for agent consumption).
  --restart            Kill any existing server and start fresh.
  --kill               Shut down the server daemon.
  --socket PATH        UNIX socket path (default: sockets/ara_agent.sock).
  --scene PATH         Scene TOML path (default: data/assets/plot/0.toml).
"""
    )
    return 0


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aractl",
        description="First-class CLI client for the Ara agent server.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of pretty-printed text",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Kill any existing server and start fresh",
    )
    parser.add_argument(
        "--kill",
        action="store_true",
        help="Shut down the server daemon",
    )
    parser.add_argument(
        "--socket",
        default=DEFAULT_SOCKET,
        help="UNIX socket path for the agent API",
    )
    parser.add_argument(
        "--scene",
        type=pathlib.Path,
        default=pathlib.Path("data/assets/plot/0.toml"),
        help="Path to the initial scene TOML file",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="LLM API key (overrides config and environment)",
    )

    sub = parser.add_subparsers(dest="cmd", help="Available subcommands")

    start_parser = sub.add_parser("start", help="Start or restart the story")
    start_parser.add_argument("--scene-id", default=None, help="Scene ID to start at")
    sub.add_parser("step", help="Advance one tick")
    sub.add_parser("next", help="Auto-step until player input is required")
    skip_parser = sub.add_parser("skip", help="Jump to a specific scene")
    skip_parser.add_argument("scene_id", help="Target scene identifier")
    sub.add_parser("state", help="Show full state snapshot")

    reply_parser = sub.add_parser("reply", help="Submit player input (or /debug)")
    reply_parser.add_argument("text", nargs="+", help="Text to submit")

    debug_parser = sub.add_parser("debug", help="Run a debug command")
    debug_parser.add_argument("command", help="Debug command name")
    debug_parser.add_argument("args", nargs="*", help="Command arguments")

    sub.add_parser("help", help="Show this message")

    args = parser.parse_args(argv)

    if args.kill:
        if _kill_server(args.socket):
            print("[agent-cli] Server shut down.")
        else:
            print("[agent-cli] No server was running.")
        return 0

    if args.restart:
        _kill_server(args.socket)

    if not args.cmd or args.cmd == "help":
        return cmd_help(args)

    handlers = {
        "start": cmd_start,
        "step": cmd_step,
        "next": cmd_next,
        "skip": cmd_skip,
        "state": cmd_state,
        "reply": cmd_reply,
        "debug": cmd_debug,
    }
    handler = handlers.get(args.cmd)
    if handler is None:
        print(f"Unknown command: {args.cmd}")
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
