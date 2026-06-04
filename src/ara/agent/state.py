"""Serialize Ara engine state into JSON-friendly dictionaries."""

from __future__ import annotations

from typing import Any

from ara.world.engine import Engine
from ara.world.story import Story
from ara.world.scene import Scene, Location, SceneChoice
from ara.world.character import Character
from ara.world.orchestrator import TurnDecision


def character_to_dict(char: Character) -> dict[str, Any]:
    """Serialize a character to a dict."""
    return {
        "id": str(char.id),
        "name": char.name,
        "importance": char.importance.name,
        "scratch": char.scratch.text,
        "card_summary": char.card_fields.get("summary", ""),
        "sprite": char.card_fields.get("sprite", ""),
        "sprites": char.sprites,
        "current_sprite": char.current_sprite,
        "crops": char.crops,
        "prev_scene_summary": char.prev_scene_summary,
    }


def location_to_dict(loc: Location) -> dict[str, Any]:
    """Serialize a location to a dict."""
    return {
        "name": loc.name,
        "description": loc.desc,
        "lore": loc.lore,
    }


def scene_choice_to_dict(choice: SceneChoice) -> dict[str, Any]:
    """Serialize a scene choice to a dict."""
    return {
        "id": choice.id,
        "description": choice.desc,
        "only_for": choice.only_for,
    }


def scene_to_dict(scene: Scene) -> dict[str, Any]:
    """Serialize a scene to a dict."""
    return {
        "id": scene.id,
        "type": scene.scene_type,
        "language": scene.language,
        "tone": scene.tone,
        "zeitgeist": scene.zeitgeist,
        "player": scene.player.name,
        "narrator": scene.narrator.name,
        "characters": [character_to_dict(c) for c in scene.character_pool],
        "locations": [location_to_dict(l) for l in scene.location_pool],
        "starting_characters": [c.name for c in scene.starting_characters],
        "starting_location": scene.starting_location.name,
        "next_choices": {
            k: scene_choice_to_dict(v) for k, v in scene.next_choices.items()
        },
    }


def decision_to_dict(dec: TurnDecision | None) -> dict[str, Any] | None:
    """Serialize a turn decision to a dict."""
    if dec is None:
        return None
    return {
        "next_char": dec.next_char.name,
        "directive": dec.directive,
        "suggestions": dec.suggestions,
        "entering_chars": [c.name for c in dec.entering_chars],
        "exiting_chars": [c.name for c in dec.exiting_chars],
        "switch_location": dec.switch_location.name if dec.switch_location else None,
        "next_scene": dec.next_scene,
        "change_sprite": dec.change_sprite,
    }


def engine_to_dict(engine: Engine) -> dict[str, Any]:
    """Serialize engine state to a dict."""
    ctx_len = 0
    if engine.ctx is not None:
        ctx_len = len(engine.ctx.context)
    return {
        "finished": engine.finished,
        "needs_player_input": engine.needs_player_input,
        "next_scene": engine.next_scene,
        "here": [c.name for c in engine.here_chars],
        "away": [c.name for c in engine.away_chars],
        "location": engine.loc.name if engine.loc else None,
        "last_decision": decision_to_dict(engine.last_decision),
        "context_length": ctx_len,
        "directives_log": {
            c.name: d for c, d in (engine.directives_log or {}).items()
        },
    }


def story_to_dict(story: Story) -> dict[str, Any]:
    """Serialize story state to a dict."""
    return {
        "finished": story.finished,
        "scene_history": story.scene_history,
        "current_scene": scene_to_dict(story.current_scene) if story.current_scene else None,
    }
