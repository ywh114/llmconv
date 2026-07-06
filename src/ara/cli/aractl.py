"""First-class CLI client for the Ara Agent API.

Usage
-----
Start / step / reply (server auto-starts on first use)::

    $ aractl start
    $ aractl step
    $ aractl next
    $ aractl reply "Hello there"
    $ aractl reply "/info"
    $ aractl debug dump
    $ aractl state
    $ aractl save 1
    $ aractl load 1
    $ aractl saves

Force a fresh server after code changes::

    $ aractl --restart step

Shut down the daemon::

    $ aractl --kill

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

from ara.agent.client import AgentClient
from ara.config import AraSettings

_settings = AraSettings()
DEFAULT_SOCKET = str(_settings.default_socket_path)
PID_FILE = _settings.sockets_path / "ara_agent_cli.pid"


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
    """Start the AgentServer as a background daemon process.

    Server stdout and stderr are appended to ``log/aractl-server.log`` so the
    daemon can be inspected and debugged without blocking the CLI.
    """
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

    log_dir = pathlib.Path("log")
    if log_dir.exists() and not log_dir.is_dir():
        backup = pathlib.Path("log.old")
        log_dir.replace(backup)
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "aractl-server.log"
    log_file = log_path.open("a", encoding="utf-8")
    log_file.write(f"\n--- server start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log_file.flush()

    # Inherit parent environment, skip HF Hub network check for cached model
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    PID_FILE.write_text(str(proc.pid))

    # Wait for server to be ready (Chroma init + embedding model load can take ~20s)
    for _ in range(300):
        if _is_server_running(socket_path):
            return
        time.sleep(0.1)
    raise RuntimeError("Server failed to start within 30 seconds")


# ------------------------------------------------------------------ #
# Client helpers
# ------------------------------------------------------------------ #


def _resolve_scene(name: str) -> pathlib.Path:
    """Convert a short scene name to a full TOML path.

    ``demo``       → ``data/assets/plot/demo/ini_scene.toml``
    ``arena``      → ``data/assets/plot/arena/ini_scene.toml``
    ``arena/mid``  → ``data/assets/plot/arena/arena_mid.toml``
    """
    if "/" in name:
        dirname, stem = name.split("/", 1)
        return pathlib.Path(f"data/assets/plot/{dirname}/{stem}.toml")
    return pathlib.Path(f"data/assets/plot/{name}/ini_scene.toml")

def _get_client(args: argparse.Namespace) -> AgentClient:
    if not _is_server_running(args.socket):
        if args.restart:
            _kill_server(args.socket)
        scene_path = args.scene_path or _resolve_scene(args.scene)
        _start_server_daemon(
            args.socket,
            scene_path,
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

    speaker = data.get("speaker")
    if speaker:
        title = data.get("speaker_title", "")
        display = f"[{title}] {speaker}" if title else speaker
        if event == "needs_player_input":
            lines.append(f"  next speaker: {display}")
        else:
            lines.append(f"  speaker: {display}")

    loc = data.get("location")
    if loc and event != "scene_loaded":
        loc_name = loc.get("id") or loc.get("name")
        if loc_name:
            lines.append(f"  location: {loc_name}")

    scene = data.get("scene")
    if scene is not None:
        lines.append(f"  scene: {scene.get('id')}")
        loc = scene.get("starting_location")
        if loc:
            lines.append(f"  location: {loc}")
        chars = scene.get("characters")
        if chars:
            names = []
            for c in chars:
                title = c.get('title', '')
                name = c.get('name', '?')
                if title:
                    names.append(f"[{title}] {name}")
                else:
                    names.append(name)
            lines.append(f"  characters: {', '.join(names)}")

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

    inner = data.get("inner", "")
    if inner.strip():
        lines.append("")
        lines.append("[inner]")
        lines.append(inner.rstrip())

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
    here = engine.get("here", [])
    if here and isinstance(here[0], dict):
        here_fmt = [f"[{c.get('title','')}] {c.get('name','?')}".strip() for c in here]
        lines.append(f"  here: {here_fmt}")
    else:
        lines.append(f"  here: {here}")
    away = engine.get("away", [])
    if away and isinstance(away[0], dict):
        away_fmt = [f"[{c.get('title','')}] {c.get('name','?')}".strip() for c in away]
        lines.append(f"  away: {away_fmt}")
    else:
        lines.append(f"  away: {away}")
    lines.append(f"  current_speaker: {engine.get('current_speaker', 'N/A')}")
    lines.append(f"  context_length: {engine.get('context_length')}")
    dec = engine.get("last_decision")
    if dec:
        title = dec.get("next_char_title", "")
        label = f"[{title}] {dec.get('next_char')}" if title else dec.get('next_char')
        lines.append(f"  last_decision: {label} - {dec.get('directive') or '(no directive)'}")
    else:
        lines.append("  last_decision: (none)")

    def _status_summary(status: dict[str, Any]) -> str:
        if not isinstance(status, dict):
            return "(untitled)"
        title = status.get("title") or "Status"
        sections = status.get("sections") or []
        parts: list[str] = []
        bars = sum(1 for s in sections if isinstance(s, dict) and s.get("type") == "bars")
        skills = sum(1 for s in sections if isinstance(s, dict) and s.get("type") == "skills")
        inv = sum(1 for s in sections if isinstance(s, dict) and s.get("type") == "inventory")
        if bars:
            parts.append(f"{bars} bar(s)")
        if skills:
            parts.append(f"{skills} skill(s)")
        if inv:
            parts.append(f"{inv} inventory")
        if parts:
            return f"{title} ({', '.join(parts)})"
        return title

    char_statuses = engine.get("character_statuses") or {}
    if char_statuses:
        lines.append("  character_statuses:")
        for name, status in char_statuses.items():
            lines.append(f"    {name}: {_status_summary(status)}")

    loc_statuses = engine.get("location_statuses") or {}
    if loc_statuses:
        lines.append("  location_statuses:")
        for name, status in loc_statuses.items():
            lines.append(f"    {name}: {_status_summary(status)}")

    sys_state = engine.get("player_status")
    if sys_state:
        lines.append(f"  player_status: {_status_summary(sys_state)}")

    free_status = engine.get("free_status")
    if free_status and isinstance(free_status, dict) and (free_status.get("title") or free_status.get("sections")):
        from ara.world.system_page import pretty_print
        lines.append("  free_status:")
        for line in pretty_print(free_status).splitlines():
            lines.append(f"    {line}")

    return "\n".join(lines)


def _fmt_char_entry(item: dict[str, Any]) -> str:
    """Format a character-like dict for display."""
    name = item.get("name", "?")
    title = item.get("title", "")
    label = f"[{title}] {name}" if title else name
    extra = {k: v for k, v in item.items() if k not in ("name", "title")}
    if extra:
        label += f" ({', '.join(f'{k}={v}' for k, v in extra.items())})"
    return label


def _fmt_debug(data: dict[str, Any]) -> str:
    lines: list[str] = ["[DEBUG]"]
    if "error" in data:
        lines.append(f"  error: {data['error']}")
        return "\n".join(lines)
    # Generic pretty-print of all keys
    char_keys = {"character", "next", "last_speaker"}
    skip_keys = {k + "_title" for k in char_keys}
    for key, value in data.items():
        if key in skip_keys:
            continue
        if key == "messages" and isinstance(value, list):
            lines.append(f"  {key}:")
            for msg in value:
                if not isinstance(msg, dict):
                    lines.append(f"    - {msg}")
                    continue
                role = msg.get("role", "?")
                name = msg.get("name", "")
                content = str(msg.get("content", "") or "")
                prefix = f"{role}/{name}" if name else role
                text = content.replace("\n", " ")[:120]
                suffix = ""
                if msg.get("tool_calls"):
                    suffix += f" [tool_calls: {len(msg['tool_calls'])}]"
                if msg.get("reasoning_content"):
                    suffix += " [reasoning]"
                lines.append(f"    [{prefix}] {text}{suffix}")
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f"  {key}:")
            for item in value:
                if isinstance(item, dict):
                    if "name" in item:
                        lines.append(f"    - {_fmt_char_entry(item)}")
                    else:
                        line = ", ".join(f"{k}={v}" for k, v in item.items())
                        lines.append(f"    - {line}")
                else:
                    lines.append(f"    - {item}")
        elif isinstance(value, dict) and "name" in value:
            lines.append(f"  {key}: {_fmt_char_entry(value)}")
        elif isinstance(value, str) and key in char_keys:
            title_key = f"{key}_title"
            title = data.get(title_key, "")
            if title:
                lines.append(f"  {key}: [{title}] {value}")
            else:
                lines.append(f"  {key}: {value}")
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


def cmd_state(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.state()
    _print(result, args)
    return 0


def _inventory_sections(player_status: dict[str, Any]) -> list[dict[str, Any]]:
    """Return inventory sections from player_status (legacy + DSL)."""
    sections: list[dict[str, Any]] = []
    if not isinstance(player_status, dict):
        return sections
    for section in player_status.get('sections', []):
        if isinstance(section, dict) and section.get('type') == 'inventory':
            sections.append(section)
    legacy = player_status.get('inventory')
    if isinstance(legacy, list) and legacy:
        sections.append({'type': 'inventory', 'items': legacy})
    return sections


def cmd_inventory(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.state()
    player_status = (
        (result.get('engine') or {}).get('player_status') or
        (result.get('story') or {}).get('player_status') or
        {}
    )
    sections = _inventory_sections(player_status)
    all_items: list[Any] = []
    for section in sections:
        items = section.get('items', [])
        if isinstance(items, list):
            all_items.extend(items)

    if args.json:
        print(_fmt_json({'items': all_items}))
        return 0

    if not all_items:
        print('[INVENTORY] (empty)')
        return 0

    print('[INVENTORY]')
    for item in all_items:
        if isinstance(item, dict):
            name = item.get('name') or item.get('label') or str(item)
            desc = item.get('description', '')
            print(f'  • {name}')
            if args.render and desc:
                print(f'    {desc}')
        else:
            print(f'  • {item}')
    return 0


def cmd_attempt(args: argparse.Namespace) -> int:
    text = " ".join(args.text)
    client = _get_client(args)
    result = client.attempt(text)
    _print(result, args)
    return 0


def cmd_reply(args: argparse.Namespace) -> int:
    text = " ".join(args.text) if args.text else ""
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
        kwargs: dict[str, Any] = {}
        if args.attempt:
            kwargs["attempt"] = args.attempt
        result = client.input(text, **kwargs)
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


def cmd_save(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.save(slot=args.slot)
    _print(result, args)
    return 0


def cmd_load(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.load(slot=args.slot)
    _print(result, args)
    return 0


def cmd_saves(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.list_saves(story_id=args.story_id or None)
    _print(result, args)
    return 0


def cmd_delete_save(args: argparse.Namespace) -> int:
    client = _get_client(args)
    result = client.delete_save(slot=args.slot, story_id=args.story_id or None)
    _print(result, args)
    return 0


def cmd_help(args: argparse.Namespace) -> int:
    print(
        """Ara Agent CLI - first-class client for the game server.

Subcommands:
  start                Start or restart the story.
  step                 Advance one tick.
  skip <scene_id>      Jump to a specific scene (abandons current scene).
  state                Show full state snapshot.
  inventory            Show player inventory (use --render for descriptions).
  reply <text>         Submit player input. If text starts with /,
                       it is treated as a debug command (first-class).
  attempt <text>       Store a pending action attempt without ending the turn.
                       Use before reply to combine both in one turn.
  debug <cmd> [args..] Run an explicit debug command.
  save <slot>          Save current state to a slot (1-99).
  load <slot>          Load state from a slot.
  saves                List all save slots.
  delete-save <slot>   Delete a save slot.
  help                 Show this message.

Global options:
  --json               Output raw JSON (for agent consumption).
  --restart            Kill any existing server and start fresh.
  --kill               Shut down the server daemon.
  --socket PATH        UNIX socket path (default: data/sockets/ara_agent.sock).
  --scene DIR          Scene dir under data/assets/plot/ (e.g. arena, arena/mid).
                       Defaults to demo (ini_scene.toml).
  --scene-path PATH    Exact path to a scene TOML (overrides --scene).
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
        default="demo",
        help="Scene dir under data/assets/plot/ (e.g. arena, arena/arena_mid). Defaults to demo (ini_scene.toml).",
    )
    parser.add_argument(
        "--scene-path",
        type=pathlib.Path,
        default=None,
        help="Exact path to a scene TOML (overrides --scene).",
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
    skip_parser = sub.add_parser("skip", help="Jump to a specific scene")
    skip_parser.add_argument("scene_id", help="Target scene identifier")
    sub.add_parser("state", help="Show full state snapshot")

    inventory_parser = sub.add_parser("inventory", help="Show player inventory")
    inventory_parser.add_argument(
        "--render",
        action="store_true",
        help="Show item descriptions inline",
    )

    reply_parser = sub.add_parser("reply", help="Submit player input (or /debug)")
    reply_parser.add_argument("text", nargs="*", help="Text to submit")
    reply_parser.add_argument("--attempt", default=None, help="Attach an action attempt")

    attempt_parser = sub.add_parser("attempt", help="Store a pending action attempt")
    attempt_parser.add_argument("text", nargs="+", help="Action attempt text")

    debug_parser = sub.add_parser("debug", help="Run a debug command")
    debug_parser.add_argument("command", help="Debug command name")
    debug_parser.add_argument("args", nargs="*", help="Command arguments")

    save_parser = sub.add_parser("save", help="Save current state to a slot")
    save_parser.add_argument("slot", type=int, default=1, nargs="?", help="Slot number (1-99)")

    load_parser = sub.add_parser("load", help="Load state from a slot")
    load_parser.add_argument("slot", type=int, default=1, nargs="?", help="Slot number")

    saves_parser = sub.add_parser("saves", help="List all save slots")
    saves_parser.add_argument("--story-id", default="", help="Filter by story ID")

    del_parser = sub.add_parser("delete-save", help="Delete a save slot")
    del_parser.add_argument("slot", type=int, help="Slot number")
    del_parser.add_argument("--story-id", default="", help="Story ID")

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
        time.sleep(0.5)  # let socket fully release before respawn

    if not args.cmd or args.cmd == "help":
        return cmd_help(args)

    handlers = {
        "start": cmd_start,
        "step": cmd_step,
        "skip": cmd_skip,
        "state": cmd_state,
        "inventory": cmd_inventory,
        "reply": cmd_reply,
        "attempt": cmd_attempt,
        "debug": cmd_debug,
        "save": cmd_save,
        "load": cmd_load,
        "saves": cmd_saves,
        "delete-save": cmd_delete_save,
    }
    handler = handlers.get(args.cmd)
    if handler is None:
        print(f"Unknown command: {args.cmd}")
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
