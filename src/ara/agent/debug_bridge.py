"""Structured debug bridge - returns dicts instead of printing to stdout.

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
                "engine, e",
                "player, p",
                "here",
                "away",
                "loc",
                "scene",
                "decision, dec",
                "scratch <name>",
                "summary <name>",
                "status <name>",
                "context <name>",
                "attempts",
                "history",
                "last",
                "exec, x <code>",
            ]
        }

    def _cmd_dump(self, args: list[str]) -> dict[str, Any]:
        return {"messages": self.state.ctx.to_list()}

    def _cmd_info(self, args: list[str]) -> dict[str, Any]:
        last_title = (
            self.state.decision.next_char.title
            if self.state.decision
            else None
        )
        return {
            "scene": self.state.scene.id,
            "location": self.state.loc.name,
            "here": [
                {"name": c.name, "title": c.title, "current_sprite": c.current_sprite}
                for c in self.state.here_chars
            ],
            "away": [
                {"name": c.name, "title": c.title, "current_sprite": c.current_sprite}
                for c in self.state.away_chars
            ],
            "context_length": len(self.state.ctx.context),
            "last_speaker": (
                self.state.decision.next_char.name
                if self.state.decision
                else None
            ),
            "last_speaker_title": last_title,
        }

    def _cmd_engine(self, args: list[str]) -> dict[str, Any]:
        engine = self.state.engine
        return {
            "needs_player_input": engine.needs_player_input,
            "turn_count": getattr(engine, "_turn_count", None),
            "speaker_history": list(getattr(engine, "_speaker_history", [])),
            "pending_attempts": list(getattr(engine, "_pending_attempts", [])),
            "finished": engine.finished,
        }

    def _cmd_player(self, args: list[str]) -> dict[str, Any]:
        player = self.state.scene.player
        return {
            "name": player.name,
            "title": player.title,
            "canonical": player.canonical_name,
            "hidden": player.hidden,
            "visible_to": list(player.visible_to),
            "importance": player.importance.name,
            "current_sprite": player.current_sprite,
        }

    def _cmd_here(self, args: list[str]) -> dict[str, Any]:
        return {
            "characters": [
                {
                    "name": c.name,
                    "title": c.title,
                    "importance": c.importance.name,
                    "current_sprite": c.current_sprite,
                }
                for c in sorted(self.state.here_chars, key=lambda x: x.name)
            ]
        }

    def _cmd_away(self, args: list[str]) -> dict[str, Any]:
        return {
            "characters": [
                {
                    "name": c.name,
                    "title": c.title,
                    "importance": c.importance.name,
                    "current_sprite": c.current_sprite,
                }
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
            "next_title": d.next_char.title,
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
        return {"character": name, "title": char.title, "scratch": char.scratch.text or "(empty)"}

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
            "title": char.title,
            "prev_scene_summary": char.prev_scene_summary or "(empty)",
        }

    def _cmd_status(self, args: list[str]) -> dict[str, Any]:
        if not args:
            return {"error": "Usage: status <character_name>"}
        name = args[0]
        char = next(
            (c for c in self.state.scene.character_pool if c.name == name),
            None,
        )
        if char is None:
            return {"error": f"Character '{name}' not found."}
        return {
            "character": name,
            "title": char.title,
            "current_sprite": char.current_sprite,
            "status": dict(char.status) if char.status else None,
        }

    def _cmd_context(self, args: list[str]) -> dict[str, Any]:
        if not args:
            return {"error": "Usage: context <character_name>"}
        name = args[0]
        char = self.state.scene.character_by_name(name)
        if char is None:
            return {"error": f"Character '{name}' not found."}
        try:
            context = self.state.engine.build_character_context(char)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "character": char.name,
            "title": char.title,
            "canonical": char.canonical_name,
            "importance": char.importance.name,
            "has_tools": char.importance != 0,
            "system_prompt": context["system_prompt"],
            "messages": context["messages"],
        }

    def _cmd_attempts(self, args: list[str]) -> dict[str, Any]:
        return {"pending_attempts": list(getattr(self.state.engine, "_pending_attempts", []))}

    def _cmd_history(self, args: list[str]) -> dict[str, Any]:
        return {"speaker_history": list(getattr(self.state.engine, "_speaker_history", []))}

    def _cmd_last(self, args: list[str]) -> dict[str, Any]:
        messages = self.state.ctx.to_list()
        if not messages:
            return {"last": None}
        return {"last": messages[-1]}

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
