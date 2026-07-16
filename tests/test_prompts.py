"""Tests for the centralized system-prompt builders in :mod:`ara.prompts`."""

from __future__ import annotations

from unittest.mock import MagicMock

from ara.memory.chroma import ChromaStore
from ara.prompts.character import character_system_prompt
from ara.prompts.narrator import narrator_system_prompt
from ara.prompts.orchestrator import orchestrator_system_prompt
from ara.prompts.summarizer import summarizer_system_prompt, summarizer_user_prompt
from ara.world.character import Importance

from tests.helpers import make_char, make_scene


def _scene():
    return make_scene("prompt_scene", MagicMock(spec=ChromaStore))


class TestNarratorPrompt:
    """narrator_system_prompt interpolates scene and character identity."""

    def test_interpolates_language_names_and_mood(self) -> None:
        scene = _scene()
        prompt = narrator_system_prompt(scene.player, scene.narrator, scene)

        assert "Reply in English only!" in prompt
        assert "You are the Narrator, the Narrator" in prompt
        assert "The player is Player." in prompt
        assert "zeitgeist: test" in prompt
        assert "tone: neutral" in prompt

    def test_language_follows_scene(self) -> None:
        scene = _scene()
        scene.language = "Chinese"
        prompt = narrator_system_prompt(scene.player, scene.narrator, scene)
        assert "Reply in Chinese only!" in prompt


class TestCharacterPrompt:
    """character_system_prompt reflects tools and importance."""

    def test_basic_prompt_has_no_tools_section(self) -> None:
        scene = _scene()
        alice = make_char("Alice", MagicMock(spec=ChromaStore))
        prompt = character_system_prompt(alice, scene)

        assert "Reply in English only!" in prompt
        assert "You are Alice." in prompt
        assert "## Available tools" not in prompt
        assert "IMPORTANT" in prompt  # default importance note

    def test_has_tools_lists_tools(self) -> None:
        scene = _scene()
        alice = make_char("Alice", MagicMock(spec=ChromaStore))
        prompt = character_system_prompt(alice, scene, has_tools=True)

        assert "## Available tools" in prompt
        assert "recall(query)" in prompt
        assert "wiki_recall(query)" in prompt
        assert "write_scratch(note)" in prompt
        assert "attempt_action(action, ...)" in prompt

    def test_anonymous_characters_are_told_they_have_no_tools(self) -> None:
        scene = _scene()
        anon = make_char(
            "Waiter", MagicMock(spec=ChromaStore), importance=Importance.ANONYMOUS
        )
        prompt = character_system_prompt(anon, scene, has_tools=False)

        assert "ANONYMOUS" in prompt
        assert "You have NO tools" in prompt

    def test_prompt_has_anonymity_rule(self) -> None:
        scene = _scene()
        alice = make_char("Alice", MagicMock(spec=ChromaStore))
        prompt = character_system_prompt(alice, scene)
        assert 'Do not assume any character is a "player"' in prompt


class TestOrchestratorPrompt:
    """orchestrator_system_prompt documents tools and identity rules."""

    def test_interpolates_names_and_mood(self) -> None:
        scene = _scene()
        prompt = orchestrator_system_prompt(scene.player, scene.narrator, scene)

        assert "Give suggestions and directives in English only!" in prompt
        assert "the role of Player" in prompt
        assert "The narrator name is Narrator." in prompt
        assert "zeitgeist of the plot is: test" in prompt
        assert "tone of the current scene is: neutral" in prompt

    def test_documents_core_tools(self) -> None:
        scene = _scene()
        prompt = orchestrator_system_prompt(scene.player, scene.narrator, scene)

        for tool_name in (
            "next_round",
            "spawn_anonymous",
            "update_status_page",
            "write_orchestrator_scratch",
            "fortune_roll",
            "fortune_random",
            "fortune_iching",
            "fortune_inspiration",
            "fortune_title",
            "fortune_ability",
            "fortune_name",
            "fortune_suite",
            "wiki_recall",
            "wiki_write",
            "wiki_forget",
        ):
            assert tool_name in prompt, f"{tool_name} missing from orchestrator prompt"

    def test_has_randomness_and_anonymity_rules(self) -> None:
        scene = _scene()
        prompt = orchestrator_system_prompt(scene.player, scene.narrator, scene)
        assert "fortune_random(distrib='normal')" in prompt
        assert 'Do not assume any character is a "player"' in prompt


class TestSummarizerSystemPrompt:
    """summarizer_system_prompt conditionally includes state notes."""

    def test_minimal_prompt_omits_optional_notes(self) -> None:
        prompt = summarizer_system_prompt(
            "English", has_changelog=False, has_player_status=False
        )
        assert "Write in English only!" in prompt
        assert "Mechanical state changes" not in prompt
        assert "Player status" not in prompt

    def test_changelog_note_included_when_changelog_present(self) -> None:
        prompt = summarizer_system_prompt(
            "English", has_changelog=True, has_player_status=False
        )
        assert "Mechanical state changes" in prompt
        assert "Player status" not in prompt

    def test_player_status_note_included_when_status_present(self) -> None:
        prompt = summarizer_system_prompt(
            "English", has_changelog=False, has_player_status=True
        )
        assert "Player status" in prompt
        assert "PLAYER_STATUS" in prompt
        assert "Mechanical state changes" not in prompt

    def test_output_format_blocks_documented(self) -> None:
        prompt = summarizer_system_prompt(
            "English", has_changelog=True, has_player_status=True
        )
        for block in (
            "SUMMARY <CharacterName>:",
            "CHARACTER <CharacterName>:",
            "ANONYMOUS <CharacterName>:",
            "LOCATION:",
            "TIME:",
            "PLAYER_STATUS:",
            "WORLD_STATUS:",
            "STATUS <CharacterName>:",
            "LOCATION_STATUS <LocationName>:",
            "SPRITE <CharacterName>:",
            "HIDDEN <CharacterName>:",
            "ORCHESTRATOR_NOTE:",
        ):
            assert block in prompt, f"{block} missing from summarizer prompt"


class TestSummarizerUserPrompt:
    """summarizer_user_prompt renders rosters, state, and defaults."""

    def _build(self, **overrides) -> str:
        scene = _scene()
        kwargs = dict(
            current_scene=scene,
            current_scene_considerations="No spoilers.",
            next_scene_plot="The party reaches the gate.",
            next_scene_considerations="",
            location_desc="A gatehouse.",
            transcript="Alice: We made it.",
            scratch_section="(none)",
            roster=["Alice", "Bob"],
            continuing=["Alice"],
            new_arrivals=["Bob"],
            mechanical_changelog=[],
            player_status={},
            world_time="dusk",
            current_character_status={},
            narrative_state={},
            next_player_name="Player",
            next_narrator_name="Narrator",
            previous_scene_characters=["Player", "Alice", "Narrator"],
        )
        kwargs.update(overrides)
        return summarizer_user_prompt(**kwargs)

    def test_rosters_and_considerations_rendered(self) -> None:
        prompt = self._build()
        assert "No spoilers." in prompt
        assert "The party reaches the gate." in prompt
        assert "Next scene cast: ['Alice', 'Bob']" in prompt
        assert "Characters continuing into next scene: ['Alice']" in prompt
        assert "Characters new to next scene: ['Bob']" in prompt
        assert "Player character: Player" in prompt
        assert "Narrator: Narrator" in prompt
        assert "A gatehouse." in prompt
        assert "World time at end of scene: dusk" in prompt

    def test_empty_state_uses_placeholders(self) -> None:
        prompt = self._build()
        assert "(No mechanical changes recorded.)" in prompt
        assert "(No player status.)" in prompt
        assert "(No stored character statuses.)" in prompt
        assert "(No narrative state.)" in prompt
        assert "(No relevant past-scene summaries.)" in prompt

    def test_changelog_and_status_blocks_rendered(self) -> None:
        prompt = self._build(
            mechanical_changelog=[{"turn": 3, "type": "set_time", "value": "night"}],
            player_status={"title": "Status", "sections": []},
            current_character_status={"Alice": {"wounded": True}},
            narrative_state={"war_declared": True},
            history_context="Scene 1: the party met.",
        )
        assert "turn 3" in prompt
        assert "'title': 'Status'" in prompt
        assert "Alice: {'wounded': True}" in prompt
        assert "war_declared" in prompt
        assert "Scene 1: the party met." in prompt

    def test_summarizer_considerations_note(self) -> None:
        prompt = self._build(summarizer_considerations="Keep it brief.")
        assert "Specific instructions for this transition:" in prompt
        assert "Keep it brief." in prompt

    def test_world_time_falls_back_to_scene_time(self) -> None:
        scene = _scene()
        scene.time = "morning"
        prompt = self._build(current_scene=scene, world_time="")
        assert "World time at end of scene: morning" in prompt
