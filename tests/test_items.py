"""Tests for plot item templates and status-page inventory integration."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.persistence.save import SaveManager, SAVE_VERSION
from ara.world.character import Character, Importance
from ara.world.engine import Engine
from ara.world.item import Item, load_item, load_item_by_id
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene
from ara.world.story import Story
from ara.world.system_page import pretty_print


def _make_char(name: str, mock_db: ChromaStore) -> Character:
    cid = uuid.uuid5(uuid.NAMESPACE_DNS, f"test.{name}")
    return Character(
        id=cid,
        canonical_name=name,
        name=name,
        card_fields={
            "name": name,
            "summary": f"{name} summary",
            "personality": f"{name} personality",
            "scenario": f"{name} scenario",
            "greeting_message": f"Hi, I'm {name}",
            "example_messages": "",
        },
        importance=Importance.IMPORTANT,
        memory=CharacterMemory(character_id=cid, db=mock_db),
        scratch=Scratchpad(),
    )


def _make_scene(scene_id: str, mock_db: ChromaStore, items: dict[str, Item] | None = None) -> Scene:
    chars = {_make_char(name, mock_db) for name in ["Player", "Narrator", "Alice", "Bob"]}
    player = next(c for c in chars if c.name == "Player")
    narrator = next(c for c in chars if c.name == "Narrator")
    loc = Location(canonical_name="room", name="room", desc="A room.")
    return Scene(
        id=scene_id,
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool=chars,
        starting_characters=chars,
        player=player,
        narrator=narrator,
        location_pool={loc},
        starting_location=loc,
        plot_considerations="",
        plot_story=f"Test {scene_id}",
        next_choices={},
        items=items or {},
    )


def _write_item_card(tmp: Path, item_id: str = "test_key") -> Path:
    items_dir = tmp / "assets" / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    path = items_dir / f"{item_id}.toml"
    path.write_text(
        f'id = "{item_id}"\n'
        f'name = "Test Key"\n'
        'description = "A key for testing."\n'
        'icon = "items/test_key"\n'
        'tags = ["key", "test"]\n\n'
        '[metadata]\n'
        'unlocks = "test_door"\n',
        encoding="utf-8",
    )
    return path


def test_load_item_card() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _write_item_card(Path(tmpdir), "test_key")
        item = load_item(path)
        assert item.id == "test_key"
        assert item.name == "Test Key"
        assert "key for testing" in item.description
        assert item.icon == "items/test_key"
        assert item.tags == ["key", "test"]
        assert item.metadata == {"unlocks": "test_door"}


def test_load_item_missing_id_raises() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.toml"
        path.write_text('name = "No ID"\n')
        with pytest.raises(ValueError, match="missing 'id'"):
            load_item(path)


def test_item_round_trip_via_dict() -> None:
    item = Item(
        id="potion",
        name="Health Potion",
        description="Restores 10 HP.",
        icon="items/potion",
        quantity=3,
        tags=["consumable"],
        metadata={"heal": 10},
    )
    restored = Item.from_dict(item.to_dict())
    assert restored == item


def test_load_item_by_id_missing_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_item_by_id(Path(tmpdir), "missing")
        assert result is None


def test_scene_load_reads_plot_items() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_item_card(tmp, "bronze_key")
        plot_dir = tmp / "assets" / "plot" / "item_test"
        plot_dir.mkdir(parents=True)
        (plot_dir / "ini_scene.toml").write_text(
            'id = "item_test"\n'
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
            'items = ["bronze_key"]\n',
            encoding="utf-8",
        )

        # Minimal character asset so Scene.load does not fail.
        cc_dir = tmp / "assets" / "cc" / "item_test"
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

        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        scene = Scene.load(plot_dir / "ini_scene.toml", db=None, config=config)
        assert "bronze_key" in scene.items
        assert scene.items["bronze_key"].name == "Test Key"


def test_update_status_page_adds_inventory_with_metadata() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)

    engine = Engine(mock_client, db=None)
    engine.start(scene)

    decision = TurnDecision(
        next_char=scene.player,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        system_changes={
            "sections": [
                {
                    "type": "inventory",
                    "items": [
                        {
                            "id": "weird_coin",
                            "name": "Weird Coin",
                            "description": "A coin from a realm you don't recognize.",
                            "metadata": {"origin": "unknown_realm"},
                        }
                    ],
                }
            ]
        },
    )
    engine._apply_decision(decision)

    inventory = engine.player_status["sections"][0]["items"]
    assert any(
        isinstance(it, dict) and it.get("id") == "weird_coin" and it.get("metadata") == {"origin": "unknown_realm"}
        for it in inventory
    )


def test_inventory_item_auto_filled_from_plot_template() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    item = Item(
        id="test_key",
        name="Test Key",
        description="A key for testing.",
        icon="items/test_key",
        tags=["key"],
        metadata={"unlocks": "test_door"},
    )
    scene = _make_scene("scene_a", mock_db, items={"test_key": item})

    engine = Engine(mock_client, db=None)
    engine.start(scene)

    decision = TurnDecision(
        next_char=scene.player,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        system_changes={
            "sections": [
                {"type": "inventory", "items": [{"id": "test_key"}]}
            ]
        },
    )
    engine._apply_decision(decision)

    entry = engine.player_status["sections"][0]["items"][0]
    assert entry["name"] == "Test Key"
    assert entry["description"] == "A key for testing."
    assert entry["metadata"] == {"unlocks": "test_door"}


def test_update_status_page_targets_free_location_and_character() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)
    alice = next(c for c in scene.character_pool if c.name == "Alice")

    engine = Engine(mock_client, db=None)
    engine.start(scene)

    # Free target
    free_decision = TurnDecision(
        next_char=scene.narrator,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        system_changes={
            "target": "free",
            "sections": [{"type": "text", "items": ["A strange omen hangs in the air."]}],
        },
    )
    engine._apply_decision(free_decision)
    assert engine.free_status["sections"][0]["items"][0] == "A strange omen hangs in the air."

    # Location target
    loc_decision = TurnDecision(
        next_char=scene.narrator,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        system_changes={
            "target": "room",
            "sections": [{"type": "inventory", "items": [{"name": "Loose Brick"}]}],
        },
    )
    engine._apply_decision(loc_decision)
    loc = scene.starting_location
    assert loc.status["sections"][0]["items"][0]["name"] == "Loose Brick"

    # Character target
    char_decision = TurnDecision(
        next_char=scene.narrator,
        directive="",
        suggestions=[],
        entering_chars=set(),
        exiting_chars=set(),
        switch_location=None,
        system_changes={
            "target": "Alice",
            "sections": [{"type": "bars", "items": [{"label": "Stress", "value": 5, "max": 10}]}],
        },
    )
    engine._apply_decision(char_decision)
    assert alice.status["sections"][0]["items"][0]["label"] == "Stress"


def test_save_load_preserves_status_pages() -> None:
    mock_db = MagicMock(spec=ChromaStore)
    mock_client = MagicMock(spec=LLMClient)
    scene = _make_scene("scene_a", mock_db)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
        (tmp / "scene_a.toml").write_text('id = "scene_a"\n')

        story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            story.start()
        story._current_scene = scene

        story.engine._player_status = {
            "title": "Player Status",
            "sections": [{"type": "inventory", "items": [{"id": "coin", "name": "Coin"}]}],
        }
        story.engine._free_status = {
            "title": "World",
            "sections": [{"type": "text", "items": ["Omen"]}],
        }
        scene.starting_location.status = {
            "title": "Room",
            "sections": [{"type": "inventory", "items": [{"name": "Brick"}]}],
        }

        manager = SaveManager(config)
        path = manager.save(story, slot=1)
        snapshot = json.loads(path.read_text())
        assert snapshot["version"] == SAVE_VERSION
        assert snapshot["engine_state"]["_free_status"]["sections"][0]["items"][0] == "Omen"
        assert snapshot["engine_state"]["_location_statuses"]["room"]["sections"][0]["items"][0]["name"] == "Brick"

        fresh_story = Story(config, mock_db, mock_client, tmp / "scene_a.toml")
        with patch("ara.world.story.Scene.load", return_value=scene):
            manager.load(fresh_story, slot=1)

        assert fresh_story.engine.free_status["sections"][0]["items"][0] == "Omen"
        assert fresh_story.engine.player_status["sections"][0]["items"][0]["name"] == "Coin"
        assert fresh_story.engine.scene.starting_location.status["sections"][0]["items"][0]["name"] == "Brick"


def test_character_status_pretty_printed_in_context() -> None:
    char = _make_char("Alice", MagicMock(spec=ChromaStore))
    char.status = {
        "title": "Alice Status",
        "sections": [
            {"type": "inventory", "items": [{"name": "Ring", "description": "A silver ring."}]},
            {"type": "bars", "items": [{"label": "HP", "value": 8, "max": 10}]},
        ],
    }
    ctx = char.status_context
    assert any("Alice Status" in m.get("content", "") for m in ctx)
    content = next(m["content"] for m in ctx if m["role"] == "assistant")
    assert "Ring: A silver ring." in content
    assert "HP" in content


def test_pretty_print_handles_strings_and_dicts() -> None:
    page = {
        "title": "Test",
        "sections": [
            {"type": "inventory", "items": ["Sword", {"name": "Shield", "description": "Heavy"}]},
        ],
    }
    text = pretty_print(page)
    assert "- Sword" in text
    assert "- Shield: Heavy" in text
