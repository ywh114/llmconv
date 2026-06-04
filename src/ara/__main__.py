"""CLI entry point for ``python -m ara``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.world.story import Story
from ara.utils.debug import DebugConsole
from ara.utils.logger import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the Ara engine from the command line.

    :param argv: Command-line arguments (defaults to :data:`sys.argv`).
    :return: Exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="ara",
        description="Multi-character AI roleplay engine",
    )
    parser.add_argument(
        "scene",
        nargs="?",
        type=Path,
        default=Path("data/assets/plot/0.toml"),
        help="Path to the initial scene TOML file (default: data/assets/plot/0.toml)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default="",
        help="LLM API key (overrides config and environment)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--debug-console",
        action="store_true",
        help="Enable interactive debug console (pauses every turn)",
    )
    args = parser.parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

    settings = AraSettings()
    if args.api_key:
        settings.api_key = args.api_key

    # Fall back to DEEPSEEK_API_KEY environment variable
    if not settings.api_key:
        settings.api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    logger.debug(f"Config loaded: endpoint={settings.api_endpoint}, model={settings.api_model}")

    if not settings.api_key:
        print(
            "Error: No API key configured. Set ARA_API_KEY, DEEPSEEK_API_KEY, "
            "or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    db = ChromaStore(settings)
    client = LLMClient(settings)
    logger.info(f"Loading story from {args.scene}")
    story = Story(settings, db, client, args.scene)

    debug_console: DebugConsole | None = None
    if args.debug_console:
        debug_console = DebugConsole(story.engine, auto_pause=True)
        story.engine.set_debug_console(debug_console)

    story.start()

    def _pause(noshell: str = "") -> None:
        if debug_console is None or story.current_scene is None:
            return
        debug_console.pause(
            scene=story.current_scene,
            ctx=story.engine.ctx,
            here_chars=story.engine.here_chars,
            away_chars=story.engine.away_chars,
            loc=story.engine.loc or story.current_scene.starting_location,
            decision=story.engine.last_decision,
            noshell=noshell,
        )

    while not story.finished:
        if (
            debug_console
            and debug_console.auto_pause
            and story._state == "running"
            and not story.engine.needs_player_input
        ):
            _pause()

        result = story.step()

        if result.event == "scene_loaded" and result.scene is not None:
            print(f"\n=== Scene: {result.scene.id} ===")
            print(f"Location: {result.scene.starting_location.name}")
            print(f"Characters: {[c.name for c in result.scene.starting_characters]}\n")

        elif result.event == "needs_player_input":
            suggestions = result.suggestions or []
            if suggestions:
                print("\n".join(suggestions))
            while True:
                try:
                    text = input(f"{story.current_scene.player.name}> ")
                except EOFError:
                    text = "[OOC: continue]"
                    break
                stripped = text.strip()
                if stripped.startswith(("/", ":")):
                    _pause(noshell=stripped[1:])
                    continue
                break
            story.submit_player_input(text)

    print("\n=== Story Complete ===")
    print(f"Scenes visited: {story.scene_history}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
