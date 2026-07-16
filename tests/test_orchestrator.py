"""Tests for :mod:`ara.world.orchestrator`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ara.llm.client import LLMClient
from ara.world.orchestrator import Orchestrator, NextRoundCapture


class TestOrchestratorScratch:
    """Tests for the orchestrator's private journal/scratchpad."""

    def test_scratch_starts_empty(self) -> None:
        """The scratchpad starts with the default placeholder."""
        orch = Orchestrator(MagicMock(spec=LLMClient), db=None)
        assert orch.scratch.text == "Nothing yet!"

    def test_write_scratch_appends_entries(self) -> None:
        """Writing entries appends them as journal notes."""
        orch = Orchestrator(MagicMock(spec=LLMClient), db=None)
        orch.write_orchestrator_scratch("Remember the portal key.")
        orch.write_orchestrator_scratch("Baylen is fast but fragile.")
        assert "[Journal]: Remember the portal key." in orch.scratch.text
        assert "[Journal]: Baylen is fast but fragile." in orch.scratch.text
        assert orch.scratch.text.count("[Journal]:") == 2

    def test_write_scratch_replaces_placeholder(self) -> None:
        """The first entry replaces the default placeholder."""
        orch = Orchestrator(MagicMock(spec=LLMClient), db=None)
        orch.write_orchestrator_scratch("First note.")
        assert orch.scratch.text == "[Journal]: First note."

    def test_prepare_for_new_scene_archives_scratch(self) -> None:
        """Scene transition archives current scratch into prev_text."""
        orch = Orchestrator(MagicMock(spec=LLMClient), db=None)
        orch.write_orchestrator_scratch("Carry this forward.")
        orch.scratch.prepare_for_new_scene()
        assert orch.scratch.prev_text == "[Journal]: Carry this forward."
        assert orch.scratch.text == "Nothing yet!"


class TestNextRoundCapture:
    """Tests for the orchestrator's next_round tool argument capture."""

    def test_end_scene_with_no_follow_up_completes_story(self) -> None:
        """If end_scene is true and there are no next scenes, accept empty next_scene."""
        capture = NextRoundCapture()
        capture.hook('{"end_scene": true, "next_scene": ""}')

        scene = MagicMock()
        scene.next_choices = {}
        scene.player = MagicMock(name="player")

        decision = capture.to_decision(set(), set(), set(), scene)

        assert decision.next_scene == ""

    def test_next_scene_auto_selected_when_empty_and_choices_exist(self) -> None:
        """If end_scene is true and next_scene is empty, auto-pick the first choice."""
        capture = NextRoundCapture()
        capture.hook('{"end_scene": true, "next_scene": ""}')

        scene = MagicMock()
        scene.next_choices = {"fin_scene": MagicMock(), "another": MagicMock()}
        scene.player = MagicMock(name="player")

        decision = capture.to_decision(set(), set(), set(), scene)

        assert decision.next_scene == "fin_scene"

    def test_invalid_next_scene_raises(self) -> None:
        """A non-empty next_scene must exist in scene.next_choices."""
        capture = NextRoundCapture()
        capture.hook('{"end_scene": true, "next_scene": "missing"}')

        scene = MagicMock()
        scene.next_choices = {"valid": MagicMock()}
        scene.player = MagicMock(name="player")

        with pytest.raises(RuntimeError):
            capture.to_decision(set(), set(), set(), scene)
