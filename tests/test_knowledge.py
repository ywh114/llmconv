from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ara.config import AraSettings
from ara.llm.client import LLMClient
from ara.llm.models import StreamResult
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.world.character import Character, Importance
from ara.world.engine import Engine
from ara.world.orchestrator import Orchestrator, TurnDecision
from ara.world.scene import Location, Scene, SceneChoice
from ara.world.setting import WorldSetting, load_world_setting
from ara.world.story import Story
from ara.world.summarizer import Summarizer


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


def _make_scene(scene_id: str, next_choices: dict, mock_db: ChromaStore) -> Scene:
    chars = {_make_char(name, mock_db) for name in ["Player", "Narrator", "NPC"]}
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
        next_choices=next_choices,
    )


class TestWorldSetting:
    """Tests for the world-setting TOML loader."""

    def test_load_world_setting(self, tmp_path: Path) -> None:
        path = tmp_path / "azur_lane.toml"
        path.write_text('''
id = "azur_lane"
name = "Azur Lane"
summary = "Alternate WWII with shipgirls."

[[factions]]
name = "Eagle Union"
description = "A maritime federation."

[[magic_systems]]
name = "Wisdom Cubes"
description = "Cores that power shipgirls."

[[history]]
period = "Great War"
event = "Sirens appeared."

[[common_sense]]
topic = "Mental Models"
fact = "Shipgirls manifest as humans."
''')
        setting = load_world_setting(path)
        assert setting.id == "azur_lane"
        assert setting.name == "Azur Lane"
        assert "shipgirls" in setting.summary
        entries = setting.wiki_entries()
        assert "world:azur_lane:summary" in entries
        assert "Eagle Union" in entries["world:azur_lane:factions:Eagle Union"]
        assert "Wisdom Cubes" in entries["world:azur_lane:magic_systems:Wisdom Cubes"]

    def test_world_setting_upserted_on_story_start(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)

        scene = _make_scene(
            "scene1",
            {"scene2": SceneChoice(id="scene2", desc="Next")},
            mock_db,
        )
        scene.world = "azur_lane"

        mock_client = MagicMock(spec=LLMClient)
        engine = Engine(mock_client, db=mock_db)
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene="scene2",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            world_dir = assets / "world"
            world_dir.mkdir(parents=True)
            (world_dir / "azur_lane.toml").write_text('''
id = "azur_lane"
name = "Azur Lane"
summary = "Alternate WWII with shipgirls."

[[factions]]
name = "Eagle Union"
description = "A maritime federation."
''')

            config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
            (tmp / "scene1.toml").write_text('id = "scene1"\n')
            story = Story(config, mock_db, mock_client, tmp / "scene1.toml")
            story.engine = engine

            scenes = {"scene1": scene, "scene2": scene}

            def fake_load(path, db, config, scene_history=None, **kwargs):
                return scenes[path.stem]

            with patch("ara.world.story.Scene.load", side_effect=fake_load):
                story.start()
                result = story.step()
                assert result.event == "scene_loaded"
                # The world setting should have been upserted into the wiki.
                upsert_calls = [
                    call for call in mock_db.upsert.call_args_list
                    if call.args[0] == "orchestrator_wiki"
                ]
                assert len(upsert_calls) == 1
                call = upsert_calls[0]
                ids = call.kwargs.get("ids", call.args[1] if len(call.args) > 1 else [])
                assert any("world:azur_lane:summary" in i for i in ids)

    def test_no_default_world_setting_when_scene_has_no_world(self) -> None:
        """If a scene does not declare a world, no arbitrary setting is loaded."""
        mock_db = MagicMock(spec=ChromaStore)

        scene = _make_scene(
            "scene1",
            {"scene2": SceneChoice(id="scene2", desc="Next")},
            mock_db,
        )
        # Intentionally leave scene.world blank.
        assert not scene.world

        mock_client = MagicMock(spec=LLMClient)
        engine = Engine(mock_client, db=mock_db)
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene="scene2",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            world_dir = assets / "world"
            world_dir.mkdir(parents=True)
            (world_dir / "dnd_forgotten_realms.toml").write_text('''
id = "dnd_forgotten_realms"
name = "Forgotten Realms"
summary = "High fantasy setting."
''')
            (world_dir / "pathfinder_golarion.toml").write_text('''
id = "pathfinder_golarion"
name = "Golarion"
summary = "Another fantasy setting."
''')

            config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
            (tmp / "scene1.toml").write_text('id = "scene1"\n')
            story = Story(config, mock_db, mock_client, tmp / "scene1.toml")
            story.engine = engine

            scenes = {"scene1": scene, "scene2": scene}

            def fake_load(path, db, config, scene_history=None, **kwargs):
                return scenes[path.stem]

            with patch("ara.world.story.Scene.load", side_effect=fake_load):
                story.start()
                story.step()
                upsert_calls = [
                    call for call in mock_db.upsert.call_args_list
                    if call.args[0] == "orchestrator_wiki"
                ]
                assert len(upsert_calls) == 0

    def test_scene_settings_upserted_on_scene_load(self) -> None:
        """A scene can declare additional setting files to load mid-story."""
        mock_db = MagicMock(spec=ChromaStore)

        scene = _make_scene(
            "scene1",
            {"scene2": SceneChoice(id="scene2", desc="Next")},
            mock_db,
        )
        scene.world = "azur_lane"
        scene.settings = ["zombie_apocalypse"]

        mock_client = MagicMock(spec=LLMClient)
        engine = Engine(mock_client, db=mock_db)
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=scene.player,
            directive="",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            next_scene="scene2",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            world_dir = assets / "world"
            world_dir.mkdir(parents=True)
            (world_dir / "azur_lane.toml").write_text('''
id = "azur_lane"
name = "Azur Lane"
summary = "Alternate WWII with shipgirls."
''')
            (world_dir / "zombie_apocalypse.toml").write_text('''
id = "zombie_apocalypse"
name = "Zombie Apocalypse"
summary = "A synthetic pathogen has turned port staff into undead."

[[factions]]
name = "Infected"
description = "Former port personnel; hostile and contagious."
''')

            config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
            (tmp / "scene1.toml").write_text('id = "scene1"\n')
            story = Story(config, mock_db, mock_client, tmp / "scene1.toml")
            story.engine = engine

            scenes = {"scene1": scene, "scene2": scene}

            def fake_load(path, db, config, scene_history=None, **kwargs):
                return scenes[path.stem]

            with patch("ara.world.story.Scene.load", side_effect=fake_load):
                story.start()
                result = story.step()
                assert result.event == "scene_loaded"
                upsert_calls = [
                    call for call in mock_db.upsert.call_args_list
                    if call.args[0] == "orchestrator_wiki"
                ]
                assert len(upsert_calls) == 2
                all_ids = []
                for call in upsert_calls:
                    ids = call.kwargs.get("ids", call.args[1] if len(call.args) > 1 else [])
                    all_ids.extend(ids)
                assert any("world:azur_lane:summary" in i for i in all_ids)
                assert any("world:zombie_apocalypse:summary" in i for i in all_ids)
                assert any("world:zombie_apocalypse:factions:Infected" in i for i in all_ids)

    def test_world_setting_not_reloaded_for_subsequent_scenes(self) -> None:
        """A world setting loaded in scene 1 should not be re-upserted in scene 2."""
        mock_db = MagicMock(spec=ChromaStore)
        scene1 = _make_scene(
            "scene1",
            {"scene2": SceneChoice(id="scene2", desc="Next")},
            mock_db,
        )
        scene1.world = "azur_lane"
        scene2 = _make_scene(
            "scene2",
            {},
            mock_db,
        )
        # scene2 has no explicit world; it inherits the already-loaded one.

        mock_client = MagicMock(spec=LLMClient)
        engine = Engine(mock_client, db=mock_db)
        engine.start = MagicMock()  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            world_dir = tmp / "assets" / "world"
            world_dir.mkdir(parents=True)
            (world_dir / "azur_lane.toml").write_text('''
id = "azur_lane"
name = "Azur Lane"
summary = "Alternate WWII with shipgirls."
''')

            config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
            (tmp / "scene1.toml").write_text('id = "scene1"\n')
            (tmp / "scene2.toml").write_text('id = "scene2"\n')
            story = Story(config, mock_db, mock_client, tmp / "scene1.toml")
            story.engine = engine

            scenes = {"scene1": scene1, "scene2": scene2}

            def fake_load(path, db, config, scene_history=None, **kwargs):
                return scenes[path.stem]

            with patch("ara.world.story.Scene.load", side_effect=fake_load):
                story.start()
                story._current_path = tmp / "scene1.toml"
                story._load_scene()
                story._current_path = tmp / "scene2.toml"
                story._load_scene()

                upsert_calls = [
                    call for call in mock_db.upsert.call_args_list
                    if call.args[0] == "orchestrator_wiki"
                ]
                assert len(upsert_calls) == 1

    def test_new_world_setting_loaded_mid_story(self) -> None:
        """A scene later in the story can declare a new world setting to load."""
        mock_db = MagicMock(spec=ChromaStore)
        scene1 = _make_scene(
            "scene1",
            {"scene2": SceneChoice(id="scene2", desc="Next")},
            mock_db,
        )
        scene1.world = "azur_lane"
        scene2 = _make_scene(
            "scene2",
            {},
            mock_db,
        )
        scene2.world = "zombie_apocalypse"

        mock_client = MagicMock(spec=LLMClient)
        engine = Engine(mock_client, db=mock_db)
        engine.start = MagicMock()  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            world_dir = tmp / "assets" / "world"
            world_dir.mkdir(parents=True)
            (world_dir / "azur_lane.toml").write_text('''
id = "azur_lane"
name = "Azur Lane"
summary = "Alternate WWII with shipgirls."
''')
            (world_dir / "zombie_apocalypse.toml").write_text('''
id = "zombie_apocalypse"
name = "Zombie Apocalypse"
summary = "A synthetic pathogen has turned port staff into undead."
''')

            config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
            (tmp / "scene1.toml").write_text('id = "scene1"\n')
            (tmp / "scene2.toml").write_text('id = "scene2"\n')
            story = Story(config, mock_db, mock_client, tmp / "scene1.toml")
            story.engine = engine

            scenes = {"scene1": scene1, "scene2": scene2}

            def fake_load(path, db, config, scene_history=None, **kwargs):
                return scenes[path.stem]

            with patch("ara.world.story.Scene.load", side_effect=fake_load):
                story.start()
                story._current_path = tmp / "scene1.toml"
                story._load_scene()
                story._current_path = tmp / "scene2.toml"
                story._load_scene()

                upsert_calls = [
                    call for call in mock_db.upsert.call_args_list
                    if call.args[0] == "orchestrator_wiki"
                ]
                assert len(upsert_calls) == 2
                all_ids = []
                for call in upsert_calls:
                    ids = call.kwargs.get("ids", call.args[1] if len(call.args) > 1 else [])
                    all_ids.extend(ids)
                assert any("world:azur_lane:summary" in i for i in all_ids)
                assert any("world:zombie_apocalypse:summary" in i for i in all_ids)

    def test_scene_setting_loaded_once_even_if_repeated(self) -> None:
        """A scene-specific setting is upserted once even if many scenes list it."""
        mock_db = MagicMock(spec=ChromaStore)
        scene1 = _make_scene(
            "scene1",
            {"scene2": SceneChoice(id="scene2", desc="Next")},
            mock_db,
        )
        scene1.settings = ["zombie_apocalypse"]
        scene2 = _make_scene(
            "scene2",
            {},
            mock_db,
        )
        scene2.settings = ["zombie_apocalypse"]

        mock_client = MagicMock(spec=LLMClient)
        engine = Engine(mock_client, db=mock_db)
        engine.start = MagicMock()  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            world_dir = tmp / "assets" / "world"
            world_dir.mkdir(parents=True)
            (world_dir / "zombie_apocalypse.toml").write_text('''
id = "zombie_apocalypse"
name = "Zombie Apocalypse"
summary = "A synthetic pathogen has turned port staff into undead."
''')

            config = AraSettings(data_dir=tmp, api_key="", api_endpoint="", api_model="")
            (tmp / "scene1.toml").write_text('id = "scene1"\n')
            (tmp / "scene2.toml").write_text('id = "scene2"\n')
            story = Story(config, mock_db, mock_client, tmp / "scene1.toml")
            story.engine = engine

            scenes = {"scene1": scene1, "scene2": scene2}

            def fake_load(path, db, config, scene_history=None, **kwargs):
                return scenes[path.stem]

            with patch("ara.world.story.Scene.load", side_effect=fake_load):
                story.start()
                story._current_path = tmp / "scene1.toml"
                story._load_scene()
                story._current_path = tmp / "scene2.toml"
                story._load_scene()

                upsert_calls = [
                    call for call in mock_db.upsert.call_args_list
                    if call.args[0] == "orchestrator_wiki"
                ]
                assert len(upsert_calls) == 1


class TestWikiTrustAndFiltering:
    """Tests for trust metadata and knowledge filtering."""

    def test_wiki_write_includes_trust_metadata(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        client = MagicMock(spec=LLMClient)
        orch = Orchestrator(client, db=mock_db)
        orch._wiki_write("topic", "content", importance="important", trust=0.8)
        mock_db.upsert.assert_called_once()
        call = mock_db.upsert.call_args
        metadatas = call.kwargs.get("metadatas", [{}])
        assert metadatas[0].get("trust") == 0.8

    def test_wiki_recall_returns_trust_annotations(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        mock_db.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"topic": "t1", "trust": 0.5}, {"topic": "t2", "trust": -1.0}]],
        }
        client = MagicMock(spec=LLMClient)
        orch = Orchestrator(client, db=mock_db)
        result = orch._wiki_recall("q", annotate_trust=True)
        assert "(trust: 0.5)" in result
        assert "(trust: -1.0)" in result


class TestSummarizerFactsAndLocations:
    """Tests for summarizer fact extraction and multi-location finalization."""

    def test_parse_response_extracts_facts(self) -> None:
        text = (
            "SUMMARY Alice:\nStill angry.\n\n"
            "FACT: The vase shattered.\n"
            "TRUST: 1.0\n"
            "SOURCE: Narrator\n\n"
            "LOCATION:\nA room with broken glass.\n"
        )
        summaries, location, time, facts, player_status_delta, character_status_updates, narrative_state, modifiers, character_overrides, anonymous_chars, orchestrator_note = Summarizer._parse_response(
            text, ["Alice"], "A room.", "day"
        )
        assert facts[0]["fact"] == "The vase shattered."
        assert not modifiers
        assert facts[0]["trust"] == 1.0
        assert facts[0]["source"] == "Narrator"
        assert "broken glass" in location
        assert player_status_delta == {}
        assert character_status_updates == {}
        assert narrative_state == {}
        assert character_overrides == {}
        assert anonymous_chars == {}
        assert orchestrator_note == ""

    def test_parse_response_extracts_player_status(self) -> None:
        text = (
            "SUMMARY Alice:\nStill angry.\n\n"
            "LOCATION:\nA room.\n\n"
            "TIME:\nevening\n\n"
            "PLAYER_STATUS:\n"
            '{"title": "Status", "sections": [{"type": "inventory", "items": ["Key"]}, {"type": "bars", "items": [{"label": "HP", "value": 90, "max": 100}]}]}\n'
        )
        summaries, location, time, facts, player_status_delta, character_status_updates, narrative_state, modifiers, character_overrides, anonymous_chars, orchestrator_note = Summarizer._parse_response(
            text, ["Alice"], "A room.", "day"
        )
        assert player_status_delta == {
            "title": "Status",
            "sections": [
                {"type": "inventory", "items": ["Key"]},
                {"type": "bars", "items": [{"label": "HP", "value": 90, "max": 100}]},
            ],
        }
        assert time == "evening"
        assert character_status_updates == {}
        assert narrative_state == {}
        assert modifiers.player_status == player_status_delta
        assert character_overrides == {}
        assert anonymous_chars == {}
        assert orchestrator_note == ""

    def test_parse_response_extracts_status_and_narrative_state(self) -> None:
        text = (
            "SUMMARY Alice:\nShe looks worried.\n\n"
            "STATUS Bob:\n{\"location\": \"infirmary\", \"wounded\": true}\n\n"
            "NARRATIVE_STATE:\n{\"zombie_outbreak\": true}\n\n"
            "LOCATION:\nA room.\n\n"
            "TIME:\nevening\n"
        )
        summaries, location, time, facts, player_status_delta, character_status_updates, narrative_state, modifiers, character_overrides, anonymous_chars, orchestrator_note = Summarizer._parse_response(
            text, ["Alice"], "A room.", "day"
        )
        assert character_status_updates == {"Bob": {"location": "infirmary", "wounded": True}}
        assert not modifiers.sprites
        assert narrative_state == {"zombie_outbreak": True}
        assert "worried" in summaries["Alice"]
        assert character_overrides == {}
        assert anonymous_chars == {}
        assert orchestrator_note == ""

    def test_parse_response_extracts_orchestrator_note(self) -> None:
        text = (
            "SUMMARY Alice:\nStill watching.\n\n"
            "ORCHESTRATOR_NOTE:\nAlice is suspicious of Bob and may act on it.\n\n"
            "LOCATION:\nA quiet room.\n"
        )
        summaries, location, time, facts, player_status_delta, character_status_updates, narrative_state, modifiers, character_overrides, anonymous_chars, orchestrator_note = Summarizer._parse_response(
            text, ["Alice"], "A room.", "day"
        )
        assert "Alice is suspicious of Bob" in orchestrator_note
        assert summaries["Alice"] == "Still watching."

    def test_finalize_locations_classifies_and_rewrites(self) -> None:
        class FakeClient:
            def complete_subagent(self, task, context, max_tokens=512):
                if "Classify" in task:
                    return "break_room: changed (ransacked)\nstorage_room: NOOP"
                return "The break room has overturned tables and shattered windows."

        summarizer = Summarizer(FakeClient())  # type: ignore[arg-type]
        descs = {
            "break_room": "A tidy break room.",
            "storage_room": "A locked storage room.",
        }
        result = summarizer._finalize_locations(
            relevant_locations=descs,
            primary_name="break_room",
            primary_desc="A ransacked break room.",
            transcript="The mob trashed the break room.",
            language="English",
        )
        assert "ransacked" in result["break_room"]
        assert result["storage_room"] == "A locked storage room."
