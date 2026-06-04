"""Structured debug bridge — returns dicts instead of printing to stdout.

Mirrors the commands from :class:`ara.utils.debug.DebugConsole` but emits
JSON-serializable dictionaries so the agent API can return them over the
socket.
"""

from __future__ import annotations

from typing import Any

from ara.utils.debug import _DebugState


class StructuredDebugBridge:
    """Execute debug commands and return structured results."""

    def __init__(self, state: _DebugState) -> None:
        self.state = state

    def run(self, command: str, args: list[str]) -> dict[str, Any]:
        """Dispatch a debug command and return a dict."""
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            # Aliases
            aliases = {
                "h": "help",
                "d": "dump",
                "i": "info",
                "x": "exec",
            }
            if command in aliases:
                handler = getattr(self, f"_cmd_{aliases[command]}", None)

        if handler:
            try:
                return handler(args)
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}"}

        return {"error": f"Unknown command: {command}. Try 'help'."}

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    def _cmd_help(self, args: list[str]) -> dict[str, Any]:
        return {
            "commands": [
                "help, h",
                "dump, d",
                "info, i",
                "here",
                "away",
                "loc",
                "scene",
                "decision, dec",
                "scratch <name>",
                "summary <name>",
                "exec, x <code>",
            ]
        }

    def _cmd_dump(self, args: list[str]) -> dict[str, Any]:
        return {"messages": self.state.ctx.to_list()}

    def _cmd_info(self, args: list[str]) -> dict[str, Any]:
        return {
            "scene": self.state.scene.id,
            "location": self.state.loc.name,
            "here": [c.name for c in self.state.here_chars],
            "away": [c.name for c in self.state.away_chars],
            "context_length": len(self.state.ctx.context),
            "last_speaker": (
                self.state.decision.next_char.name
                if self.state.decision
                else None
            ),
        }

    def _cmd_here(self, args: list[str]) -> dict[str, Any]:
        return {
            "characters": [
                {"name": c.name, "importance": c.importance.name}
                for c in sorted(self.state.here_chars, key=lambda x: x.name)
            ]
        }

    def _cmd_away(self, args: list[str]) -> dict[str, Any]:
        return {
            "characters": [
                {"name": c.name, "importance": c.importance.name}
                for c in sorted(self.state.away_chars, key=lambda x: x.name)
            ]
        }

    def _cmd_loc(self, args: list[str]) -> dict[str, Any]:
        return {
            "name": self.state.loc.name,
            "description": self.state.loc.desc,
            "lore": self.state.loc.lore,
        }

    def _cmd_scene(self, args: list[str]) -> dict[str, Any]:
        sc = self.state.scene
        return {
            "id": sc.id,
            "language": sc.language,
            "tone": sc.tone,
            "zeitgeist": sc.zeitgeist,
            "locations": [l.name for l in sc.location_pool],
            "next_choices": list(sc.next_choices.keys()),
        }

    def _cmd_decision(self, args: list[str]) -> dict[str, Any]:
        d = self.state.decision
        if d is None:
            return {"error": "No decision yet."}
        return {
            "next": d.next_char.name,
            "directive": d.directive or None,
            "suggestions": d.suggestions or None,
            "enter": [c.name for c in d.entering_chars],
            "exit": [c.name for c in d.exiting_chars],
            "switch_location": d.switch_location.name if d.switch_location else None,
            "next_scene": d.next_scene,
        }

    def _cmd_scratch(self, args: list[str]) -> dict[str, Any]:
        if not args:
            return {"error": "Usage: scratch <character_name>"}
        name = args[0]
        char = next(
            (c for c in self.state.scene.character_pool if c.name == name),
            None,
        )
        if char is None:
            return {"error": f"Character '{name}' not found."}
        return {"character": name, "scratch": char.scratch.text or "(empty)"}

    def _cmd_summary(self, args: list[str]) -> dict[str, Any]:
        if not args:
            return {"error": "Usage: summary <character_name>"}
        name = args[0]
        char = next(
            (c for c in self.state.scene.character_pool if c.name == name),
            None,
        )
        if char is None:
            return {"error": f"Character '{name}' not found."}
        return {
            "character": name,
            "prev_scene_summary": char.prev_scene_summary or "(empty)",
        }

    def _cmd_exec(self, args: list[str]) -> dict[str, Any]:
        code = " ".join(args)
        if not code:
            return {"error": "Usage: exec <python_code>"}
        _locals = {
            "engine": self.state.engine,
            "scene": self.state.scene,
            "ctx": self.state.ctx,
            "here": self.state.here_chars,
            "away": self.state.away_chars,
            "loc": self.state.loc,
            "decision": self.state.decision,
            "client": self.state.engine.client,
            "orchestrator": self.state.engine.orchestrator,
        }
        try:
            exec(code, _locals)
            return {"result": "Executed."}
        except Exception as exc:
            return {"error": f"Execution error: {exc}"}
