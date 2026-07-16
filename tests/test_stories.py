"""Tests for story discovery and story serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock


from ara.agent.state import story_to_dict
from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.webclient.stories import discover_stories
from ara.world.story import Story


def test_discover_stories_includes_opening_text() -> None:
    """discover_stories surfaces the opening_text field from ini_scene.toml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_dir = Path(tmpdir)
        story_dir = plot_dir / "test_story"
        story_dir.mkdir()
        (story_dir / "ini_scene.toml").write_text(
            'title = "Test"\n'
            'opening_text = "Hello, world!"\n'
            'first_scene = "scene1"\n'
        )

        stories = discover_stories(plot_dir)
        assert len(stories) == 1
        assert stories[0]["opening_text"] == "Hello, world!"


def test_story_to_dict_includes_opening_text() -> None:
    """story_to_dict carries opening_text read from ini_scene.toml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_dir = Path(tmpdir)
        ini = plot_dir / "ini_scene.toml"
        ini.write_text(
            'title = "Test"\n'
            'opening_text = "Attribution line."\n'
            'first_scene = "scene1"\n'
        )
        (plot_dir / "scene1.toml").write_text('id = "scene1"\n')

        config = AraSettings(data_dir=Path(tmpdir), api_key="", api_endpoint="", api_model="")
        story = Story(config, MagicMock(spec=ChromaStore), MagicMock(spec=LLMClient), ini)
        data = story_to_dict(story)
        assert data["opening_text"] == "Attribution line."
