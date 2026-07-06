"""Tests for per-story asset resolution with global fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ara.config import AraSettings
from ara.world.fortune import load_hexagrams, load_inspiration
from ara.world.item import Item, load_item_by_id
from ara.world.scene import Scene
from ara.world.setting import resolve_world_setting_path


def _make_assets(tmp: Path) -> AraSettings:
    """Populate a temp data tree with global and per-story asset overrides."""
    (tmp / "assets" / "world").mkdir(parents=True)
    (tmp / "assets" / "world" / "shared.toml").write_text(
        'id = "shared"\nsummary = "global"\n', encoding="utf-8"
    )
    (tmp / "assets" / "world" / "demo").mkdir()
    (tmp / "assets" / "world" / "demo" / "shared.toml").write_text(
        'id = "shared"\nsummary = "story"\n', encoding="utf-8"
    )

    (tmp / "assets" / "fortune").mkdir(parents=True)
    (tmp / "assets" / "fortune" / "iching.json").write_text(
        '[{"number": 1, "name": "global"}]', encoding="utf-8"
    )
    (tmp / "assets" / "fortune" / "demo").mkdir()
    (tmp / "assets" / "fortune" / "demo" / "iching.json").write_text(
        '[{"number": 2, "name": "story"}]', encoding="utf-8"
    )
    (tmp / "assets" / "fortune" / "inspiration.json").write_text(
        '["global"]', encoding="utf-8"
    )
    (tmp / "assets" / "fortune" / "demo" / "inspiration.json").write_text(
        '["story"]', encoding="utf-8"
    )

    (tmp / "assets" / "items").mkdir(parents=True)
    (tmp / "assets" / "items" / "gem.toml").write_text(
        'id = "gem"\nname = "Global Gem"\n', encoding="utf-8"
    )
    (tmp / "assets" / "items" / "demo").mkdir()
    (tmp / "assets" / "items" / "demo" / "gem.toml").write_text(
        'id = "gem"\nname = "Story Gem"\n', encoding="utf-8"
    )

    return AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")


def _make_character_dirs(tmp: Path, story: str) -> None:
    cc_dir = tmp / "assets" / "cc" / story
    for name in ("Player", "Narrator"):
        d = cc_dir / name
        d.mkdir(parents=True)
        (d / "card.toml").write_text(
            f'name = "{name}"\n'
            'summary = ""\n'
            'personality = ""\n'
            'scenario = ""\n'
            'greeting_message = ""\n'
            'example_messages = ""\n',
            encoding="utf-8",
        )


def test_resolve_world_setting_prefers_story_override() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _make_assets(tmp)
        global_path = resolve_world_setting_path("shared", config)
        story_path = resolve_world_setting_path("shared", config, story="demo")
        assert story_path == config.world_path("demo") / "shared.toml"
        assert story_path != global_path
        assert story_path.read_text(encoding="utf-8") == 'id = "shared"\nsummary = "story"\n'


def test_load_hexagrams_prefers_story_override() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_assets(Path(tmpdir))
        global_hex = load_hexagrams(config=config)
        story_hex = load_hexagrams(story="demo", config=config)
        assert global_hex[0]["name"] == "global"
        assert story_hex[0]["name"] == "story"


def test_load_inspiration_falls_back_to_global() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _make_assets(tmp)
        (tmp / "assets" / "fortune" / "demo" / "inspiration.json").unlink()
        words = load_inspiration(story="demo", config=config)
        assert words == ["global"]


def test_load_item_by_id_prefers_story_override() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_assets(Path(tmpdir))
        global_item = load_item_by_id(config.data_dir, "gem")
        story_item = load_item_by_id(config.data_dir, "gem", story="demo")
        assert isinstance(global_item, Item)
        assert isinstance(story_item, Item)
        assert global_item.name == "Global Gem"
        assert story_item.name == "Story Gem"


def test_scene_load_prefers_story_item() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = _make_assets(tmp)
        _make_character_dirs(tmp, "demo")
        plot_dir = tmp / "assets" / "plot" / "demo"
        plot_dir.mkdir(parents=True)
        (plot_dir / "ini_scene.toml").write_text(
            'id = "demo"\n'
            'language = "English"\n'
            'zeitgeist = "test"\n'
            'tone = "neutral"\n\n'
            '[character]\n'
            'pool = ["Player", "Narrator"]\n'
            'inits = ["Player", "Narrator"]\n'
            'player = "Player"\n'
            'narrator = "Narrator"\n\n'
            '[location]\n'
            'pool = ["room"]\n'
            'init = "room"\n\n'
            '[location.descs]\n'
            'room = "A room."\n\n'
            '[plot]\n'
            'items = ["gem"]\n',
            encoding="utf-8",
        )

        scene = Scene.load(plot_dir / "ini_scene.toml", db=None, config=config)
        assert "gem" in scene.items
        assert scene.items["gem"].name == "Story Gem"
