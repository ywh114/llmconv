"""Tests for importance system integration and anonymous background characters."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ara.config import AraSettings
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, NullMemory, Scratchpad
from ara.models import GameRole, Importance, StreamResult
from ara.world.character import Character, create_anonymous_character, load_character
from ara.world.engine import Engine
from ara.world.orchestrator import TurnDecision
from ara.world.scene import Location, Scene, SceneChoice
from ara.world.story import _merge_characters


def _make_char(name: str, importance: Importance, mock_db: ChromaStore) -> Character:
    """Build a Character with the given importance."""
    cid = uuid.uuid5(uuid.NAMESPACE_DNS, f"test.{name}")
    return Character(
        id=cid,
        name=name,
        card_fields={
            "name": name,
            "summary": f"{name} summary",
            "personality": f"{name} personality",
            "scenario": f"{name} scenario",
            "greeting_message": f"Hi, I'm {name}",
            "example_messages": "",
        },
        importance=importance,
        memory=CharacterMemory(character_id=cid, db=mock_db),
        scratch=Scratchpad(),
    )


def _make_scene_with_chars(chars: list[Character]) -> Scene:
    """Build a minimal scene containing the given characters."""
    player = next(c for c in chars if c.name == "Player")
    narrator = next(c for c in chars if c.name == "Narrator")
    loc = Location(name="room", desc="A room.")
    return Scene(
        id="test",
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool=set(chars),
        starting_characters=set(chars),
        player=player,
        narrator=narrator,
        location_pool={loc},
        starting_location=loc,
        plot_considerations="",
        plot_story="Test scene",
        next_choices={},
    )


class TestAnonymousCharacterCreation:
    """Anonymous characters are created when asset directories are missing."""

    def test_create_anonymous_character(self) -> None:
        """Factory should produce an ANONYMOUS character with NullMemory."""
        char = create_anonymous_character("Waiter")
        assert char.name == "Waiter"
        assert char.importance == Importance.ANONYMOUS
        assert char.card_fields["summary"] == ""
        assert isinstance(char.memory, NullMemory)

    def test_scene_load_errors_for_missing_dir_without_anonymous(self) -> None:
        """A missing asset dir without [anonymous] entry should raise RuntimeError."""
        mock_db = MagicMock(spec=ChromaStore)
        config = AraSettings()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            cc = assets / "cc"
            plot = assets / "plot"
            cc.mkdir(parents=True)
            plot.mkdir(parents=True)

            for name in ["Player", "Narrator"]:
                d = cc / name
                d.mkdir()
                (d / "card.toml").write_text(f'name = "{name}"\n')

            toml = plot / "test.toml"
            toml.write_text('''
id = "test"
language = "English"
zeitgeist = "test"
tone = "neutral"

[character]
pool = ["Player", "Narrator", "Waiter"]
inits = ["Player", "Narrator", "Waiter"]
player = "Player"
narrator = "Narrator"

[location]
pool = ["room"]
init = "room"

[location.descs]
room = "A room."

[plot]
considerations = ""
scene = "Test"

[plot.next]
considerations = "None"
''')
            config.data_dir = tmp
            with pytest.raises(RuntimeError, match=r"not defined in \[anonymous\]"):
                Scene.load(toml, mock_db, config)

    def test_scene_load_creates_anonymous_for_missing_dir(self) -> None:
        """Scene.load should auto-create anonymous chars when asset dir is missing."""
        mock_db = MagicMock(spec=ChromaStore)
        config = AraSettings()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            assets = tmp / "assets"
            cc = assets / "cc"
            plot = assets / "plot"
            cc.mkdir(parents=True)
            plot.mkdir(parents=True)

            # Only create asset dir for Player and Narrator
            for name in ["Player", "Narrator"]:
                d = cc / name
                d.mkdir()
                (d / "card.toml").write_text(f'name = "{name}"\n')

            toml = plot / "test.toml"
            toml.write_text('''
id = "test"
language = "English"
zeitgeist = "test"
tone = "neutral"

[character]
pool = ["Player", "Narrator", "Waiter"]
inits = ["Player", "Narrator", "Waiter"]
player = "Player"
narrator = "Narrator"

[anonymous]
Waiter = "A background waiter who serves drinks."

[location]
pool = ["room"]
init = "room"

[location.descs]
room = "A room."

[plot]
considerations = ""
scene = "Test"

[plot.next]
considerations = "None"
''')
            config.data_dir = tmp
            scene = Scene.load(toml, mock_db, config)

        assert len(scene.character_pool) == 3
        waiter = next(c for c in scene.character_pool if c.name == "Waiter")
        assert waiter.importance == Importance.ANONYMOUS
        assert isinstance(waiter.memory, NullMemory)


class TestAnonymousEngineBehavior:
    """ANONYMOUS characters skip tools and memory persistence."""

    def test_anonymous_character_gets_no_tools(self) -> None:
        """An ANONYMOUS NPC should not be offered recall/think/write_scratch tools."""
        mock_db = MagicMock(spec=ChromaStore)
        anon = _make_char("Waiter", Importance.ANONYMOUS, mock_db)
        player = _make_char("Player", Importance.EIGEN, mock_db)
        narrator = _make_char("Narrator", Importance.IMPORTANT, mock_db)
        scene = _make_scene_with_chars([player, narrator, anon])

        calls: list[dict] = []

        class _FakeClient:
            def complete(self, **kwargs):
                calls.append(kwargs)
                return StreamResult(content="Here's your coffee.")

            def complete_subagent(self, **kwargs):
                return ""

        engine = Engine(_FakeClient(), db=mock_db)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=anon,
            directive="Serve coffee",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            edit_location="",
            next_scene=None,
        )

        engine.start(scene)
        engine.step()

        # The LLM call should not have been given any tools
        assert len(calls) == 1
        assert calls[0].get("tools") is None

    def test_anonymous_character_does_not_store_memory(self) -> None:
        """After speaking, an ANONYMOUS character should not call add_conversation."""
        mock_db = MagicMock(spec=ChromaStore)
        anon = _make_char("Waiter", Importance.ANONYMOUS, mock_db)
        player = _make_char("Player", Importance.EIGEN, mock_db)
        narrator = _make_char("Narrator", Importance.IMPORTANT, mock_db)
        scene = _make_scene_with_chars([player, narrator, anon])

        class _FakeClient:
            def complete(self, **kwargs):
                return StreamResult(content="Here's your coffee.")

            def complete_subagent(self, **kwargs):
                return ""

        engine = Engine(_FakeClient(), db=mock_db)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=anon,
            directive="Serve coffee",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            edit_location="",
            next_scene=None,
        )

        engine.start(scene)
        engine.step()

        assert anon.memory.db.upsert.call_count == 0  # type: ignore[union-attr]

    def test_important_character_still_gets_tools_and_memory(self) -> None:
        """IMPORTANT characters must keep full tool access and memory storage."""
        mock_db = MagicMock(spec=ChromaStore)
        npc = _make_char("NPC", Importance.IMPORTANT, mock_db)
        player = _make_char("Player", Importance.EIGEN, mock_db)
        narrator = _make_char("Narrator", Importance.IMPORTANT, mock_db)
        scene = _make_scene_with_chars([player, narrator, npc])

        calls: list[dict] = []

        class _FakeClient:
            def complete(self, **kwargs):
                calls.append(kwargs)
                return StreamResult(content="I remember that.")

            def complete_subagent(self, **kwargs):
                return ""

        engine = Engine(_FakeClient(), db=mock_db)  # type: ignore[arg-type]
        engine.orchestrator = MagicMock()
        engine.orchestrator.decide_next_turn.return_value = TurnDecision(
            next_char=npc,
            directive="Recall something",
            suggestions=[],
            entering_chars=set(),
            exiting_chars=set(),
            switch_location=None,
            edit_location="",
            next_scene=None,
        )

        engine.start(scene)
        engine.step()

        assert calls[-1]["tools"] is not None
        assert len(calls[-1]["tools"]) == 3
        assert npc.memory.db.upsert.call_count >= 1  # type: ignore[union-attr]


class TestAnonymousStoryMerge:
    """Anonymous characters should not carry state across scenes."""

    def test_merge_skips_anonymous(self) -> None:
        """_merge_characters should skip ANONYMOUS characters."""
        mock_db = MagicMock(spec=ChromaStore)

        player = _make_char("Player", Importance.EIGEN, mock_db)
        narrator = _make_char("Narrator", Importance.IMPORTANT, mock_db)

        old_important = _make_char("NPC", Importance.IMPORTANT, mock_db)
        old_important.scratch.text = "Important memory"

        old_anon = _make_char("Waiter", Importance.ANONYMOUS, mock_db)
        old_anon.scratch.text = "Anon memory"

        new_important = _make_char("NPC", Importance.IMPORTANT, mock_db)
        new_anon = _make_char("Waiter", Importance.ANONYMOUS, mock_db)

        prev = _make_scene_with_chars([player, narrator, old_important, old_anon])
        nxt = _make_scene_with_chars([player, narrator, new_important, new_anon])

        _merge_characters(prev, nxt)

        # Scratchpads are NOT carried over — only memory is
        assert new_important.scratch.text == "Nothing yet!"
        assert new_anon.scratch.text == "Nothing yet!"
