"""Tests for the location asset system."""

from __future__ import annotations

from pathlib import Path


from ara.config import AraSettings
from ara.world.scene import Location, load_location


def test_load_location_inline(tmp_path: Path) -> None:
    """A location with no asset directory uses inline data and the lc/ path."""
    loc = load_location(
        tmp_path / "does_not_exist",
        name="building_break_room",
        inline_desc="A break room.",
        inline_loading="break_room_bg",
    )
    assert loc.name == "building_break_room"
    assert loc.desc == "A break room."
    assert loc.loading_background == "break_room_bg"
    assert loc.backgrounds == ["break_room_bg"]
    assert loc.current_background == "break_room_bg"
    assert loc.asset_dir is None
    assert loc.background_url() == "lc/building_break_room/break_room_bg.png"


def test_load_location_asset(tmp_path: Path) -> None:
    """A location asset directory is parsed into backgrounds and metadata."""
    loc_dir = tmp_path / "kitchen"
    loc_dir.mkdir()
    (loc_dir / "card.toml").write_text(
        'id = "kitchen"\nname = "Kitchen"\n'
        'description = "A small galley kitchen."\n'
        'lore = "Used by the port office staff."\n'
        'loading_background = "kitchen_loading"\n',
        encoding="utf-8",
    )
    (loc_dir / "meta.toml").write_text(
        'backgrounds = ["counter_day", "counter_night"]\n'
        'default_background = "counter_night"\n\n'
        '[counter_day]\n'
        'focus = [[100, 100, 50]]\n',
        encoding="utf-8",
    )
    # Create dummy image files so discovery also works.
    (loc_dir / "counter_day.png").write_bytes(b"")
    (loc_dir / "counter_night.png").write_bytes(b"")

    loc = load_location(
        loc_dir,
        name="kitchen",
        inline_desc="",
        inline_loading="",
    )
    assert loc.name == "kitchen"
    assert loc.desc == "A small galley kitchen."
    assert loc.lore == "Used by the port office staff."
    assert loc.loading_background == "kitchen_loading"
    assert loc.backgrounds == ["counter_day", "counter_night"]
    assert loc.current_background == "counter_night"
    assert loc.asset_dir == loc_dir
    assert loc.background_url() == "lc/kitchen/counter_night.png"
    assert "counter_day" in loc.background_crops


def test_load_location_inline_overrides_card(tmp_path: Path) -> None:
    """Inline scene TOML values override card.toml defaults."""
    loc_dir = tmp_path / "room"
    loc_dir.mkdir()
    (loc_dir / "card.toml").write_text(
        'description = "From card."\n', encoding="utf-8"
    )
    (loc_dir / "room_day.png").write_bytes(b"")

    loc = load_location(
        loc_dir,
        name="room",
        inline_desc="From inline.",
        inline_loading="",
    )
    assert loc.desc == "From inline."
    assert loc.backgrounds == ["room_day"]


def test_location_background_switch() -> None:
    """Background switching updates the active background and URL."""
    loc = Location(
        canonical_name="kitchen",
        name="kitchen",
        desc="A kitchen.",
        backgrounds=["counter_day", "counter_night"],
        current_background="counter_day",
        asset_dir=Path("/fake/kitchen"),
    )
    assert loc.background_url() == "lc/kitchen/counter_day.png"
    loc.current_background = "counter_night"
    assert loc.background_url() == "lc/kitchen/counter_night.png"


def test_scene_load_uses_location_assets(tmp_path: Path) -> None:
    """Scene.load resolves locations from lc/ assets when present."""
    from ara.memory.chroma import ChromaStore
    from ara.world.scene import Scene
    from unittest.mock import MagicMock

    # Build a minimal character asset so Scene.load does not fail.
    cc_dir = tmp_path / "assets" / "cc" / "test_loc" / "Player"
    cc_dir.mkdir(parents=True)
    (cc_dir / "card.toml").write_text(
        'name = "Player"\nsummary = ""\npersonality = ""\n'
        'scenario = ""\ngreeting_message = ""\nexample_messages = ""\n',
        encoding="utf-8",
    )

    lc_dir = tmp_path / "assets" / "lc" / "test_loc" / "test_room"
    lc_dir.mkdir(parents=True)
    (lc_dir / "card.toml").write_text(
        'description = "A test room from asset."\n', encoding="utf-8"
    )
    (lc_dir / "test_room_day.png").write_bytes(b"")

    scene_dir = tmp_path / "assets" / "plot" / "test_loc"
    scene_dir.mkdir(parents=True)
    scene_file = scene_dir / "scene.toml"
    scene_file.write_text(
        'id = "test_loc"\n'
        '[character]\n'
        'pool = ["Player"]\n'
        'inits = ["Player"]\n'
        'player = "Player"\n'
        'narrator = "Player"\n'
        '[location]\n'
        'pool = ["test_room"]\n'
        'init = "test_room"\n',
        encoding="utf-8",
    )

    config = AraSettings(
        data_dir=tmp_path,
        api_key="",
        api_endpoint="",
        api_model="",
    )
    db = MagicMock(spec=ChromaStore)
    scene = Scene.load(scene_file, db, config)
    assert scene.starting_location.desc == "A test room from asset."
    assert scene.starting_location.backgrounds == ["test_room_day"]
    assert scene.starting_location.background_url() == "lc/test_loc/test_room/test_room_day.png"
