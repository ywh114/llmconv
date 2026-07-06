"""Serialize Ara engine state into JSON-friendly dictionaries."""

from __future__ import annotations

from typing import Any

from ara.world.engine import Engine
from ara.world.story import Story
from ara.world.scene import Scene, Location, SceneChoice
from ara.world.character import Character
from ara.world.orchestrator import TurnDecision
from ara.world.system_page import SystemPage


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
        "sprite_descriptions": char.sprite_descriptions,
        "prev_scene_summary": char.prev_scene_summary,
        "names": dict(char.names),
        "title": char.title,
        "hidden": char.hidden,
        "visible_to": list(char.visible_to),
        "inner_log": list(char.inner_log),
        "status": dict(char.status),
    }


def location_to_dict(loc: Location) -> dict[str, Any]:
    """Serialize a location to a dict."""
    return {
        "name": loc.name,
        "description": loc.desc,
        "lore": loc.lore,
        "backgrounds": loc.backgrounds,
        "current_background": loc.current_background,
        "background_url": loc.background_url(),
        "loading_background": loc.loading_background,
        "status": dict(loc.status),
    }


def scene_choice_to_dict(choice: SceneChoice) -> dict[str, Any]:
    """Serialize a scene choice to a dict."""
    return {
        "id": choice.id,
        "description": choice.desc,
        "prereq_scenes": choice.prereq_scenes,
    }


def scene_to_dict(scene: Scene) -> dict[str, Any]:
    """Serialize a scene to a dict."""
    return {
        "id": scene.id,
        "name": scene.name,
        "asset_story_name": scene.asset_story_name,
        "type": scene.scene_type,
        "language": scene.language,
        "tone": scene.tone,
        "zeitgeist": scene.zeitgeist,
        "time": scene.time,
        "world_map": scene.world_map,
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
        "next_char_title": dec.next_char.title,
        "directive": dec.directive,
        "suggestions": dec.suggestions,
        "entering_chars": [c.name for c in dec.entering_chars],
        "exiting_chars": [c.name for c in dec.exiting_chars],
        "switch_location": dec.switch_location.name if dec.switch_location else None,
        "next_scene": dec.next_scene,
        "change_sprite": dec.change_sprite,
        "switch_background": dec.switch_background,
        "spawn_anonymous": dec.spawn_anonymous,
        "set_time": dec.set_time,
        "system_changes": dec.system_changes,
        "response_mode": dec.response_mode,
    }


def engine_to_dict(engine: Engine) -> dict[str, Any]:
    """Serialize engine state to a dict."""
    ctx_len = 0
    if engine.ctx is not None:
        ctx_len = len(engine.ctx.context)
    speaker_history = getattr(engine, "_speaker_history", [])
    current_speaker = speaker_history[-1] if speaker_history else None
    return {
        "finished": engine.finished,
        "next_scene": engine.next_scene,
        "here": [{"name": c.name, "title": c.title} for c in engine.here_chars],
        "away": [{"name": c.name, "title": c.title} for c in engine.away_chars],
        "location": engine.loc.name if engine.loc else None,
        "world_time": engine.world_time,
        "current_speaker": current_speaker,
        "last_decision": decision_to_dict(engine.last_decision),
        "context_length": ctx_len,
        "directives_log": {
            c.name: d for c, d in (engine.directives_log or {}).items()
        },
        "player_status": engine.player_status,
        "free_status": engine.free_status,
        "character_statuses": {
            c.name: SystemPage.from_dict(c.status).to_dict()
            for c in (engine.scene.character_pool if engine.scene else [])
        },
        "location_statuses": {
            loc.name: SystemPage.from_dict(loc.status).to_dict()
            for loc in (engine.scene.location_pool if engine.scene else [])
        },
    }


def build_visual_state(story: Story) -> dict[str, Any]:
    """Build the full visual state the browser needs to render the scene.

    Includes the current scene, engine state, reconstructed dialogue history,
    and which characters are currently present.  Used by ``/start``, ``/load``
    and ``/reset`` so the browser can initialise its UI without waiting for a
    ``scene_loaded`` event.
    """
    engine = story.engine
    scene = story.current_scene

    # Reconstruct visible history from the conversation context.
    history: list[dict[str, str]] = []
    if engine.ctx is not None and scene is not None:
        player_name = scene.player.name
        for msg in engine.ctx.context:
            role = msg.get("role")
            name = msg.get("name", "")
            content = msg.get("content", "") or ""
            if role == "assistant" and content:
                history.append({"speaker": name, "text": content})
            elif role == "user" and name == player_name and content:
                history.append({"speaker": "Player", "text": content})

    # Determine who is currently present.  After a fresh start the engine has
    # not been started yet, so fall back to the scene's starting characters.
    here = list(engine.here_chars) if engine.here_chars else list(scene.starting_characters) if scene else []
    away = list(engine.away_chars) if engine.away_chars else []

    engine_dict = engine_to_dict(engine)
    return {
        # Backward-compatible story fields
        "finished": story.finished,
        "scene_history": story.scene_history,
        "current_scene": scene_to_dict(scene) if scene else None,
        # Full visual state for browser init
        "scene": scene_to_dict(scene) if scene else None,
        "engine": engine_dict,
        "history": history,
        "current_speaker": engine_dict.get("current_speaker"),
        "here": [character_to_dict(c) for c in here],
        "away": [character_to_dict(c) for c in away],
        "location": location_to_dict(engine.loc) if engine.loc else (
            location_to_dict(scene.starting_location) if scene else None
        ),
    }


def story_to_dict(story: Story) -> dict[str, Any]:
    """Serialize story state to a dict."""
    return {
        "finished": story.finished,
        "scene_history": story.scene_history,
        "current_scene": scene_to_dict(story.current_scene) if story.current_scene else None,
        "opening_text": story._story_meta.get("opening_text", ""),
    }
