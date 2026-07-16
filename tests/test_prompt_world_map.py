"""Tests for the open-world textual map on scenes."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from ara.config import AraSettings
from ara.memory.chroma import ChromaStore
from ara.world.scene import Scene


def test_scene_load_world_map() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        story = "world_map_test"
        cc = tmp / "assets" / "cc" / story
        plot = tmp / "assets" / "plot" / story
        plot.mkdir(parents=True)
        for name in ("Player", "Narrator"):
            path = cc / name
            path.mkdir(parents=True)
            (path / "card.toml").write_text(
                f'name = "{name}"\n'
                'summary = ""\n'
                'personality = ""\n'
                'scenario = ""\n'
                'greeting_message = ""\n'
                'example_messages = ""\n',
                encoding="utf-8",
            )

        scene_toml = plot / "world_map_scene.toml"
        scene_toml.write_text(
            'id = "world_map_scene"\n'
            'name = "World Map Scene"\n'
            'language = "English"\n'
            '\n'
            '[character]\n'
            'pool = ["Player", "Narrator"]\n'
            'inits = ["Player", "Narrator"]\n'
            'player = "Player"\n'
            'narrator = "Narrator"\n'
            '\n'
            '[location]\n'
            'pool = ["room"]\n'
            'init = "room"\n'
            '\n'
            '[location.descs]\n'
            'room = "A room."\n'
            '\n'
            '[world]\n'
            'map = """\n'
            '玄城位于东州中部，北接云雾谷。\n'
            '"""\n'
            '\n'
            '[plot]\n'
            'scene = "Test"\n',
            encoding="utf-8",
        )

        mock_db = MagicMock(spec=ChromaStore)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        scene = Scene.load(scene_toml, mock_db, config)

        assert "玄城" in scene.world_map
        assert "World map" in scene.plot_as_tool_content()
