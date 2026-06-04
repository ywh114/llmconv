"""Minimal demo script that wires the engine together and starts a story.

This is equivalent to running ``python -m ara`` but can be customised
programmatically (e.g. to inject a mock LLM client for testing).
"""

from __future__ import annotations

from pathlib import Path

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.world.story import Story


def run_demo(scene_path: Path | None = None) -> None:
    """Load settings, initialise storage, and run a story.

    :param scene_path: Override path to the initial scene TOML file.
        Defaults to ``data/assets/plot/0.toml`` relative to the working
        directory.
    """
    settings = AraSettings()
    db = ChromaStore(settings)
    client = LLMClient(settings)

    if scene_path is None:
        scene_path = Path("data/assets/plot/0.toml")

    story = Story(settings, db, client, scene_path)
    story.run()


if __name__ == "__main__":
    run_demo()
