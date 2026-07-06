"""Tests for canonical (scripted) mode."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, Scratchpad
from ara.world.character import Importance
from ara.world.character import Character
from ara.world.engine import Engine, _hidden_not_visible
from ara.world.scene import Location, Scene, SceneChoice, _validate_canonical_events


def _make_char(name: str, db: MagicMock) -> Character:
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
        importance=1,
        memory=CharacterMemory(character_id=cid, db=db),
        scratch=Scratchpad(),
    )


def _make_scene(**kwargs) -> Scene:
    mock_db = MagicMock(spec=ChromaStore)
    player = _make_char("Player", mock_db)
    narrator = _make_char("Narrator", mock_db)
    npc = _make_char("NPC", mock_db)
    loc = Location(canonical_name="room", name="room", desc="A room.")

    defaults = dict(
        id="test",
        language="English",
        zeitgeist="test",
        tone="neutral",
        scene_type="normal",
        character_pool={player, narrator, npc},
        starting_characters={player, narrator, npc},
        player=player,
        narrator=narrator,
        location_pool={loc},
        starting_location=loc,
        plot_considerations="",
        plot_story="Test scene",
        next_choices={},
    )
    defaults.update(kwargs)
    return Scene(**defaults)


class TestCanonicalReplay:
    """Basic scripted event replay without AI."""

    def test_turns_replay_in_order(self) -> None:
        scene = _make_scene(
            canonical_events=[
                {"event": "turn", "speaker": "Narrator", "output": "Hello."},
                {"event": "turn", "speaker": "NPC", "output": "Hi there."},
                {"event": "turn", "speaker": "Player", "output": "Hey."},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        r1 = engine.step()
        assert r1.speaker == "Narrator"
        r2 = engine.step()
        assert r2.speaker == "NPC"
        r3 = engine.step()
        assert r3.speaker == "Player"

    def test_auto_submit_player_does_not_wait(self) -> None:
        scene = _make_scene(
            canonical_events=[
                {"event": "turn", "speaker": "Player", "output": "I do something."},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        result = engine.step()
        assert result.speaker == "Player"
        assert not engine.needs_player_input

    def test_enter_exit_tracked(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        scene = Scene(
            id="test",
            language="English",
            zeitgeist="test",
            tone="neutral",
            scene_type="normal",
            character_pool={player, narrator, npc},
            starting_characters={player, narrator},  # NPC away
            player=player,
            narrator=narrator,
            location_pool={loc},
            starting_location=loc,
            plot_considerations="",
            plot_story="Test scene",
            next_choices={},
            canonical_events=[
                {"event": "turn", "speaker": "Narrator", "output": "NPC enters.", "enter": ["NPC"]},
                {"event": "turn", "speaker": "NPC", "output": "Hello.", "exit": ["NPC"]},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        assert "NPC" not in {c.name for c in engine.here_chars}
        engine.step()
        assert "NPC" in {c.name for c in engine.here_chars}
        engine.step()
        assert "NPC" not in {c.name for c in engine.here_chars}

    def test_player_can_start_away_and_enter(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        scene = Scene(
            id="test",
            language="English",
            zeitgeist="test",
            tone="neutral",
            scene_type="normal",
            character_pool={player, narrator, npc},
            starting_characters={narrator},  # Player away
            player=player,
            narrator=narrator,
            location_pool={loc},
            starting_location=loc,
            plot_considerations="",
            plot_story="Test scene",
            next_choices={},
            canonical_events=[
                {"event": "turn", "speaker": "Narrator", "output": "You arrive.", "enter": ["Player"]},
                {"event": "turn", "speaker": "Player", "output": "I'm here."},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        assert "Player" not in {c.name for c in engine.here_chars}
        engine.step()
        assert "Player" in {c.name for c in engine.here_chars}

    def test_narrator_can_exit(self) -> None:
        scene = _make_scene(
            canonical_events=[
                {"event": "turn", "speaker": "Narrator", "output": "I leave.", "exit": ["Narrator"]},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        assert "Narrator" in {c.name for c in engine.here_chars}
        engine.step()
        assert "Narrator" not in {c.name for c in engine.here_chars}

    def test_scene_ended_without_choices(self) -> None:
        scene = _make_scene(
            next_choices={"end": SceneChoice(id="end", desc="End")},
            canonical_events=[
                {"event": "turn", "speaker": "Narrator", "output": "Bye."},
                {"event": "scene_ended", "next_scene": "end"},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        engine.step()
        result = engine.step()
        assert result.scene_ended
        assert result.next_scene == "end"
        assert engine.finished

    def test_story_complete_event(self) -> None:
        scene = _make_scene(
            canonical_events=[
                {"event": "turn", "speaker": "Narrator", "output": "The end."},
                {"event": "story_complete"},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        engine.step()
        result = engine.step()
        assert result.scene_ended
        assert result.next_scene is None
        assert engine.finished


class TestCanonicalChoices:
    """End-of-scene player choices in canonical mode."""

    def test_choices_return_needs_player_input(self) -> None:
        scene = _make_scene(
            next_choices={
                "apple": SceneChoice(id="apple", desc="Apple scene"),
                "pear": SceneChoice(id="pear", desc="Pear scene"),
            },
            canonical_events=[
                {"event": "turn", "speaker": "Narrator", "output": "Choose."},
                {
                    "event": "scene_ended",
                    "choices": [
                        {"hint": "Buy apples", "text": "I want apples.", "next_scene": "apple"},
                        {"hint": "Buy pears", "text": "I want pears.", "next_scene": "pear"},
                    ],
                },
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        engine.step()  # Narrator turn
        result = engine.step()  # Scene ended with choices

        assert result.needs_player_input
        assert result.suggestions == ["Buy apples", "Buy pears"]
        assert engine.needs_player_input

    def test_choice_text_sets_next_scene(self) -> None:
        scene = _make_scene(
            next_choices={
                "apple": SceneChoice(id="apple", desc="Apple scene"),
                "pear": SceneChoice(id="pear", desc="Pear scene"),
            },
            canonical_events=[
                {"event": "turn", "speaker": "Narrator", "output": "Choose."},
                {
                    "event": "scene_ended",
                    "choices": [
                        {"hint": "Buy apples", "text": "I want apples.", "next_scene": "apple"},
                        {"hint": "Buy pears", "text": "I want pears.", "next_scene": "pear"},
                    ],
                },
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        engine.step()
        engine.step()
        engine.submit_player_input("I want pears.")

        assert engine.next_scene == "pear"
        assert not engine.needs_player_input

    def test_choice_submits_to_context(self) -> None:
        scene = _make_scene(
            next_choices={
                "apple": SceneChoice(id="apple", desc="Apple scene"),
            },
            canonical_events=[
                {
                    "event": "scene_ended",
                    "choices": [
                        {"hint": "Buy apples", "text": "I want apples.", "next_scene": "apple"},
                    ],
                },
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        engine.step()
        engine.submit_player_input("I want apples.")

        messages = engine.ctx.to_list()
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert any("I want apples." in m.get("content", "") for m in user_msgs)

    def test_invalid_choice_raises(self) -> None:
        scene = _make_scene(
            next_choices={
                "apple": SceneChoice(id="apple", desc="Apple scene"),
            },
            canonical_events=[
                {
                    "event": "scene_ended",
                    "choices": [
                        {"hint": "Buy apples", "text": "I want apples.", "next_scene": "apple"},
                    ],
                },
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        engine.step()
        with pytest.raises(RuntimeError, match="Invalid canonical choice"):
            engine.submit_player_input("I want bananas.")

    def test_step_after_choice_ends_scene(self) -> None:
        scene = _make_scene(
            next_choices={
                "apple": SceneChoice(id="apple", desc="Apple scene"),
            },
            canonical_events=[
                {
                    "event": "scene_ended",
                    "choices": [
                        {"hint": "Buy apples", "text": "I want apples.", "next_scene": "apple"},
                    ],
                },
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        engine.step()
        engine.submit_player_input("I want apples.")
        result = engine.step()

        assert result.scene_ended
        assert result.next_scene == "apple"
        assert engine.finished


class TestCanonicalValidation:
    """Validation of canonical JSON scripts."""

    def test_unknown_event_type_raises(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        with pytest.raises(RuntimeError, match="unknown event type"):
            _validate_canonical_events(
                {"events": [{"event": "unknown", "speaker": "Narrator", "output": "Oops."}]},
                {player, narrator, npc},
                {player, narrator, npc},
                {loc},
                {},
                player,
                narrator,
            )

    def test_double_enter_raises(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        with pytest.raises(RuntimeError, match="already here"):
            _validate_canonical_events(
                {"events": [
                    {"event": "turn", "speaker": "Narrator", "output": "Enter.", "enter": ["NPC"]},
                    {"event": "turn", "speaker": "Narrator", "output": "Enter again.", "enter": ["NPC"]},
                ]},
                {player, narrator, npc},
                {player, narrator},  # NPC away
                {loc},
                {},
                player,
                narrator,
            )

    def test_double_exit_raises(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        with pytest.raises(RuntimeError, match="already away"):
            _validate_canonical_events(
                {"events": [
                    {"event": "turn", "speaker": "Narrator", "output": "Exit.", "exit": ["NPC"]},
                    {"event": "turn", "speaker": "Narrator", "output": "Exit again.", "exit": ["NPC"]},
                ]},
                {player, narrator, npc},
                {player, narrator, npc},
                {loc},
                {},
                player,
                narrator,
            )

    def test_choices_only_on_last_event(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        with pytest.raises(RuntimeError, match="only allowed on the last event"):
            _validate_canonical_events(
                {"events": [
                    {
                        "event": "scene_ended",
                        "choices": [
                            {"hint": "End", "text": "End.", "next_scene": "end"},
                        ],
                    },
                    {"event": "turn", "speaker": "Narrator", "output": "Oops."},
                ]},
                {player, narrator, npc},
                {player, narrator, npc},
                {loc},
                {"end": SceneChoice(id="end", desc="End")},
                player,
                narrator,
            )

    def test_choices_and_next_scene_conflict(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        with pytest.raises(RuntimeError, match="cannot have both"):
            _validate_canonical_events(
                {"events": [
                    {
                        "event": "scene_ended",
                        "next_scene": "end",
                        "choices": [
                            {"hint": "End", "text": "End.", "next_scene": "end"},
                        ],
                    },
                ]},
                {player, narrator, npc},
                {player, narrator, npc},
                {loc},
                {"end": SceneChoice(id="end", desc="End")},
                player,
                narrator,
            )

    def test_choice_missing_hint(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        with pytest.raises(RuntimeError, match="hint"):
            _validate_canonical_events(
                {"events": [
                    {
                        "event": "scene_ended",
                        "choices": [
                            {"text": "End.", "next_scene": "end"},
                        ],
                    },
                ]},
                {player, narrator, npc},
                {player, narrator, npc},
                {loc},
                {"end": SceneChoice(id="end", desc="End")},
                player,
                narrator,
            )

    def test_choice_missing_text(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        with pytest.raises(RuntimeError, match="text"):
            _validate_canonical_events(
                {"events": [
                    {
                        "event": "scene_ended",
                        "choices": [
                            {"hint": "End", "next_scene": "end"},
                        ],
                    },
                ]},
                {player, narrator, npc},
                {player, narrator, npc},
                {loc},
                {"end": SceneChoice(id="end", desc="End")},
                player,
                narrator,
            )

    def test_choice_invalid_next_scene(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = _make_char("Player", mock_db)
        narrator = _make_char("Narrator", mock_db)
        npc = _make_char("NPC", mock_db)
        loc = Location(canonical_name="room", name="room", desc="A room.")
        with pytest.raises(RuntimeError, match="not in next_choices"):
            _validate_canonical_events(
                {"events": [
                    {
                        "event": "scene_ended",
                        "choices": [
                            {"hint": "End", "text": "End.", "next_scene": "invalid"},
                        ],
                    },
                ]},
                {player, narrator, npc},
                {player, narrator, npc},
                {loc},
                {"end": SceneChoice(id="end", desc="End")},
                player,
                narrator,
            )


class TestCanonicalLocalizedHidden:
    """Hidden visibility uses canonical IDs even with localized display names."""

    def _make_localized_char(self, canonical: str, display: str, db: MagicMock) -> Character:
        cid = uuid.uuid5(uuid.NAMESPACE_DNS, f"test.{canonical}")
        return Character(
            id=cid,
            canonical_name=canonical,
            name=display,
            card_fields={
                "name": display,
                "summary": f"{display} summary",
                "personality": f"{display} personality",
                "scenario": f"{display} scenario",
                "greeting_message": f"Hi, I'm {display}",
                "example_messages": "",
            },
            importance=Importance.IMPORTANT,
            memory=CharacterMemory(character_id=cid, db=db),
            scratch=Scratchpad(),
        )

    def test_hidden_not_visible_uses_canonical_ids(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        yeqingxuan = self._make_localized_char("yeqingxuan", "叶清宣", mock_db)
        luwei = self._make_localized_char("luwei", "陆薇", mock_db)
        luwei.hidden = True
        luwei.visible_to = {"yeqingxuan"}

        hidden = _hidden_not_visible(yeqingxuan, {yeqingxuan, luwei})
        assert hidden == set()

        stranger = self._make_localized_char("stranger", "路人", mock_db)
        hidden = _hidden_not_visible(stranger, {yeqingxuan, luwei, stranger})
        assert hidden == {"luwei"}

    def test_canonical_replay_resolves_localized_display_names(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = self._make_localized_char("player", "玩家", mock_db)
        narrator = self._make_localized_char("narrator", "旁白", mock_db)
        yeqingxuan = self._make_localized_char("yeqingxuan", "叶清宣", mock_db)
        luwei = self._make_localized_char("luwei", "陆薇", mock_db)
        loc = Location(canonical_name="room", name="房间", desc="A room.")

        luwei.hidden = True
        luwei.visible_to = {"yeqingxuan"}

        scene = Scene(
            id="test",
            language="Chinese",
            zeitgeist="test",
            tone="neutral",
            scene_type="normal",
            character_pool={player, narrator, yeqingxuan, luwei},
            starting_characters={player, narrator, yeqingxuan, luwei},
            player=player,
            narrator=narrator,
            location_pool={loc},
            starting_location=loc,
            plot_considerations="",
            plot_story="Test scene",
            next_choices={},
            canonical_events=[
                {"event": "turn", "speaker": "陆薇", "output": "Secret."},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)

        result = engine.step()
        assert result.speaker == "luwei"
        assert result.output == "Secret."

        # The engine stores context messages with canonical metadata.
        hidden_msgs = [m for m in engine.ctx.context if m.get("_hidden")]
        assert len(hidden_msgs) == 1
        assert hidden_msgs[0].get("_canonical_name") == "luwei"
        assert hidden_msgs[0].get("name") == "陆薇"

    def test_localized_hidden_message_filters_by_canonical_observer(self) -> None:
        mock_db = MagicMock(spec=ChromaStore)
        player = self._make_localized_char("player", "玩家", mock_db)
        narrator = self._make_localized_char("narrator", "旁白", mock_db)
        yeqingxuan = self._make_localized_char("yeqingxuan", "叶清宣", mock_db)
        luwei = self._make_localized_char("luwei", "陆薇", mock_db)
        loc = Location(canonical_name="room", name="房间", desc="A room.")

        luwei.hidden = True
        luwei.visible_to = {"yeqingxuan"}

        scene = Scene(
            id="test",
            language="Chinese",
            zeitgeist="test",
            tone="neutral",
            scene_type="normal",
            character_pool={player, narrator, yeqingxuan, luwei},
            starting_characters={player, narrator, yeqingxuan, luwei},
            player=player,
            narrator=narrator,
            location_pool={loc},
            starting_location=loc,
            plot_considerations="",
            plot_story="Test scene",
            next_choices={},
            canonical_events=[
                {"event": "turn", "speaker": "luwei", "output": "Secret."},
            ],
        )
        engine = Engine(MagicMock())  # type: ignore[arg-type]
        engine.start(scene)
        engine.step()

        yeq_branch = engine.ctx.branch()
        yeq_branch.filter_to("yeqingxuan")
        assert any("Secret." in m.get("content", "") for m in yeq_branch.context)

        player_branch = engine.ctx.branch()
        player_branch.filter_to("player")
        assert not any("Secret." in m.get("content", "") for m in player_branch.context)
