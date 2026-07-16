"""Interactive debug console for the Ara engine.

Inspired by the old project's ``get_user_debug`` hook, the console provides
real-time introspection into the conversation state, LLM context, and engine
internals.  It can be triggered automatically every turn or on-demand via
``/command`` or ``:command`` at the player prompt.
"""

from __future__ import annotations

import pprint

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ara.world.engine import Engine
    from ara.world.scene import Scene, Location
    from ara.world.character import Character
    from ara.llm.context import ConversationContext


class DebugConsole:
    """Lightweight REPL for inspecting engine state during a scene.

    :param engine: The :class:`Engine` instance driving the scene.
    :param auto_pause: When ``True``, the console opens automatically before
        every turn.  When ``False``, it is only invoked on-demand.
    """

    def __init__(self, engine: Engine, auto_pause: bool = False) -> None:
        """Create a debug console.

        :param engine: The :class:`Engine` instance driving the scene.
        :param auto_pause: When ``True``, the console opens automatically before
            every turn.  When ``False``, it is only invoked on-demand.
        """
        self.engine = engine
        self.auto_pause = auto_pause

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def pause(
        self,
        scene: Scene,
        ctx: ConversationContext,
        here_chars: set[Character],
        away_chars: set[Character],
        loc: Location,
        decision: Any | None = None,
        noshell: str = "",
    ) -> None:
        """Enter the debug console.

        :param scene: Current scene.
        :param ctx: Conversation context.
        :param here_chars: Characters present in the scene.
        :param away_chars: Characters away from the scene.
        :param loc: Current location.
        :param decision: The most recent orchestrator decision, if any.
        :param noshell: When non-empty, execute a single command and return
            without entering the interactive REPL.
        """
        state = _DebugState(
            console=self,
            engine=self.engine,
            scene=scene,
            ctx=ctx,
            here_chars=here_chars,
            away_chars=away_chars,
            loc=loc,
            decision=decision,
        )

        if noshell:
            self._run_one(state, noshell)
            return

        print("\n[Debug console - type 'help' for commands, 'exit' to resume]")
        while True:
            try:
                raw = input("Debug> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not raw:
                continue
            if raw.lower() in ("exit", "quit", "q"):
                break
            self._run_one(state, raw)

    def _run_one(self, state: _DebugState, cmdline: str) -> None:
        """Execute a single debug command."""
        parts = cmdline.split()
        cmd = parts[0].lower()
        args = parts[1:]

        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            # Aliases
            aliases = {
                "h": "help",
                "d": "dump",
                "i": "info",
                "x": "exec",
            }
            if cmd in aliases:
                handler = getattr(self, f"_cmd_{aliases[cmd]}", None)

        if handler:
            try:
                handler(state, args)
            except Exception as exc:
                print(f"Error: {exc}")
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")

    # ------------------------------------------------------------------ #
    # Commands
    # ------------------------------------------------------------------ #

    def _cmd_help(self, state: _DebugState, args: list[str]) -> None:
        """Show help text."""
        print(
            """Debug console commands:
  help, h           Show this message
  dump, d           Pretty-print the full LLM conversation context
  info, i           Show engine state summary
  here              List characters present in the scene
  away              List characters away from the scene
  loc               Show current location details
  scene             Show scene metadata
  decision, dec     Show last orchestrator decision
  scratch <name>    Show a character's scratchpad
  summary <name>    Show a character's prev-scene summary
  exec, x <code>    Execute arbitrary Python code (DANGEROUS)
  exit, q           Resume the game
"""
        )

    def _cmd_dump(self, state: _DebugState, args: list[str]) -> None:
        """Pretty-print the conversation context."""
        pprint.pprint(state.ctx.to_list(), width=120)

    def _cmd_info(self, state: _DebugState, args: list[str]) -> None:
        """Show engine state summary."""
        print(f"Scene: {state.scene.id}")
        print(f"Location: {state.loc.name}")
        print(f"Here: {[c.display_name_with_title() for c in state.here_chars]}")
        print(f"Away: {[c.display_name_with_title() for c in state.away_chars]}")
        print(f"Context length: {len(state.ctx.context)} messages")
        print(f"Last speaker: {state.decision.next_char.display_name_with_title() if state.decision else 'N/A'}")

    def _cmd_here(self, state: _DebugState, args: list[str]) -> None:
        """List present characters."""
        for c in sorted(state.here_chars, key=lambda x: x.name):
            print(f"  {c.display_name_with_title()} - {c.importance.name}")

    def _cmd_away(self, state: _DebugState, args: list[str]) -> None:
        """List away characters."""
        for c in sorted(state.away_chars, key=lambda x: x.name):
            print(f"  {c.display_name_with_title()} - {c.importance.name}")

    def _cmd_loc(self, state: _DebugState, args: list[str]) -> None:
        """Show current location."""
        print(f"Name: {state.loc.name}")
        print(f"Description: {state.loc.desc}")
        print(f"Lore: {state.loc.lore}")

    def _cmd_scene(self, state: _DebugState, args: list[str]) -> None:
        """Show scene metadata."""
        print(f"ID: {state.scene.id}")
        print(f"Language: {state.scene.language}")
        print(f"Tone: {state.scene.tone}")
        print(f"Zeitgeist: {state.scene.zeitgeist}")
        print(f"Locations: {[l.name for l in state.scene.location_pool]}")
        print(f"Next choices: {list(state.scene.next_choices.keys())}")

    def _cmd_decision(self, state: _DebugState, args: list[str]) -> None:
        """Show last orchestrator decision."""
        if state.decision is None:
            print("No decision yet.")
            return
        d = state.decision
        print(f"Next: {d.next_char.name}")
        print(f"Directive: {d.directive or '(none)'}")
        print(f"Suggestions: {d.suggestions or '(none)'}")
        print(f"Enter: {[c.name for c in d.entering_chars]}")
        print(f"Exit: {[c.name for c in d.exiting_chars]}")
        print(f"Switch location: {d.switch_location.name if d.switch_location else '(none)'}")
        print(f"Next scene: {d.next_scene or '(none)'}")

    def _cmd_scratch(self, state: _DebugState, args: list[str]) -> None:
        """Show a character's scratchpad."""
        if not args:
            print("Usage: scratch <character_name>")
            return
        name = args[0]
        char = next(
            (c for c in state.scene.character_pool if c.name == name),
            None,
        )
        if char is None:
            print(f"Character '{name}' not found.")
            return
        print(f"--- {name}'s scratchpad ---")
        print(char.scratch.text or "(empty)")

    def _cmd_summary(self, state: _DebugState, args: list[str]) -> None:
        """Show a character's prev-scene summary."""
        if not args:
            print("Usage: summary <character_name>")
            return
        name = args[0]
        char = next(
            (c for c in state.scene.character_pool if c.name == name),
            None,
        )
        if char is None:
            print(f"Character '{name}' not found.")
            return
        print(f"--- {name}'s prev-scene summary ---")
        print(char.prev_scene_summary or "(empty)")

    def _cmd_exec(self, state: _DebugState, args: list[str]) -> None:
        """Execute arbitrary Python code."""
        code = " ".join(args)
        if not code:
            print("Usage: exec <python_code>")
            return
        # Provide convenient locals
        _locals = {
            "engine": state.engine,
            "scene": state.scene,
            "ctx": state.ctx,
            "here": state.here_chars,
            "away": state.away_chars,
            "loc": state.loc,
            "decision": state.decision,
            "client": state.engine.client,
            "orchestrator": state.engine.orchestrator,
        }
        try:
            exec(code, _locals)
        except Exception as exc:
            print(f"Execution error: {exc}")


class _DebugState:
    """Snapshot of engine state passed to debug commands."""

    def __init__(
        self,
        console: DebugConsole,
        engine: Engine,
        scene: Scene,
        ctx: ConversationContext,
        here_chars: set[Any],
        away_chars: set[Any],
        loc: Location,
        decision: Any | None,
    ) -> None:
        """Capture a snapshot of engine state for debug commands."""
        self.console = console
        self.engine = engine
        self.scene = scene
        self.ctx = ctx
        self.here_chars = here_chars
        self.away_chars = away_chars
        self.loc = loc
        self.decision = decision
