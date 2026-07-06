"""Scene definition and TOML loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import tomllib

from ara.config import AraSettings
from ara.memory.chroma import ChromaStore
from ara.world.character import Character, create_anonymous_character, load_character
from ara.world.item import Item, load_item
from ara.world.i18n import normalize_language
from ara.world.registry import AssetRegistry
from ara.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Location:
    """A place within a scene.

    :ivar canonical_name: Stable latin identifier (usually the asset directory name).
    :ivar name: Localized display name.
    :ivar desc: Human-readable description used in narrator prompts.
    :ivar lore: Optional extended background information.
    :ivar loading_background: Optional filename stem for a loading-screen image.
    :ivar backgrounds: List of available background image stems for this location.
    :ivar current_background: The active background stem; rendered by the frontend.
    :ivar background_crops: Optional crop/focus data per background stem.
    :ivar asset_dir: Path to the location asset directory, if loaded from ``lc/``.
    """

    canonical_name: str
    name: str
    desc: str
    lore: str = ""
    loading_background: str = ""
    backgrounds: list[str] = field(default_factory=list)
    current_background: str = ""
    background_crops: dict[str, dict] = field(default_factory=dict)
    asset_dir: Path | None = None
    status: dict[str, Any] = field(default_factory=dict)
    """System-page DSL status for items/state present at this location."""
    names: dict[str, str] = field(default_factory=dict)
    """Alternative language names for the location (display-name aliases)."""

    def display_name(self, language: str | None = None) -> str:
        """Return the name to use for *language*.

        Falls back to the canonical name when no alias exists.
        """
        if language and language != "en":
            return self.names.get(language, self.name)
        return self.name

    def __hash__(self) -> int:
        """Hash based on canonical location name."""
        return hash(self.canonical_name)

    def __eq__(self, other: object) -> bool:
        """Equality is canonical-name-based."""
        if not isinstance(other, Location):
            return NotImplemented
        return self.canonical_name == other.canonical_name

    def background_url(self) -> str:
        """Return the active background asset path segment.

        Locations use ``lc/<story>/<canonical_name>/<background>.png`` when
        loaded from a per-story asset directory, otherwise fall back to
        ``lc/<canonical_name>/<background>.png``.
        """
        story = ""
        if self.asset_dir:
            parts = self.asset_dir.parts
            # Per-story layout: .../lc/<story>/<location>/
            if len(parts) >= 3 and parts[-3] == "lc":
                story = parts[-2] + "/"
        bg = self.current_background or self.loading_background
        return f"lc/{story}{self.canonical_name}/{bg}.png"
    def scratch_context(self, system_name: str = "System") -> list[dict]:
        """Return a context snippet describing this location.

        :param system_name: Name attribute for the system messages.
        :return: Two-message context (user request + assistant description).
        """
        return [
            {
                "role": "user",
                "content": "The following is a description of your surroundings.",
                "name": system_name,
            },
            {
                "role": "assistant",
                "content": f"Description:\n{self.desc}\nLore:\n{self.lore}",
                "name": system_name,
            },
        ]


def load_location(
    path: Path,
    name: str = "",
    inline_desc: str = "",
    inline_loading: str = "",
    language: str = "en",
) -> Location:
    """Load a location asset from ``data/assets/lc/<story>/<name>/``.

    Falls back to inline description/loading data when no asset directory exists.

    :param path: Directory containing ``card.toml`` and optional ``meta.toml``.
    :param name: Location identifier used in the scene. If empty, derived from
        the card's ``[names]`` table.
    :param inline_desc: Description from the scene TOML ``[location.descs]`` table.
    :param inline_loading: Loading background from ``[location.loading]``.
    :param language: Preferred language for the display name.
    :return: Initialised :class:`Location`.
    """
    card_path = path / "card.toml"
    meta_path = path / "meta.toml"

    card: dict[str, Any] = {}
    if card_path.exists():
        with card_path.open("rb") as f:
            card = tomllib.load(f)

    desc = inline_desc or card.get("description", "")
    lore = card.get("lore", "")
    loading_background = inline_loading or card.get("loading_background", "")

    # The directory name is the canonical ID when an asset exists; otherwise
    # the caller-supplied name is the canonical ID.  The card ``name`` field
    # and ``[names]`` table provide localized display names.
    canonical = path.name if card_path.exists() else (name or path.name)
    display = card.get("name", "") or canonical

    names: dict[str, str] = {}
    raw_names = card.get("names", {})
    if isinstance(raw_names, dict):
        names = {str(k): str(v) for k, v in raw_names.items() if isinstance(v, str)}
    if canonical and "en" not in names:
        names["en"] = canonical
    if display and language not in names:
        names[language] = display
    if not name:
        name = names.get(language, names.get("en", canonical))

    backgrounds: list[str] = []
    current_background = ""
    background_crops: dict[str, dict] = {}

    if meta_path.exists():
        with meta_path.open("rb") as f:
            meta = tomllib.load(f)
        explicit = meta.get("backgrounds", [])
        if explicit:
            backgrounds = list(explicit)
        else:
            # Auto-discover PNG stems, ignoring the loading background if present.
            backgrounds = sorted(
                {p.stem for p in path.glob("*.png") if p.is_file()}
            )
        current_background = meta.get("default_background", backgrounds[0] if backgrounds else "")
        for bg in backgrounds:
            bg_meta = meta.get(bg, {})
            if isinstance(bg_meta, dict):
                background_crops[bg] = {
                    k: v
                    for k, v in bg_meta.items()
                    if k in ("topleft", "bottomright", "center", "focus")
                }
    else:
        # No meta.toml: auto-discover image files.
        backgrounds = sorted({p.stem for p in path.glob("*.png") if p.is_file()})
        if backgrounds:
            current_background = backgrounds[0]

    # If no backgrounds were discovered but an inline loading background exists,
    # keep the legacy single-background behaviour.
    if not backgrounds and loading_background:
        backgrounds = [loading_background]
        current_background = loading_background

    return Location(
        canonical_name=canonical,
        name=name,
        desc=desc,
        lore=lore,
        loading_background=loading_background,
        backgrounds=backgrounds,
        current_background=current_background,
        background_crops=background_crops,
        asset_dir=path if card_path.exists() else None,
        status={},
        names=names,
    )


def _resolve_by_name(
    name: str,
    by_canonical: dict[str, Any],
    by_display: dict[str, Any] | None = None,
) -> Any | None:
    """Resolve *name* against canonical names, then optional display names."""
    if name in by_canonical:
        return by_canonical[name]
    if by_display and name in by_display:
        return by_display[name]
    return None


def _validate_canonical_events(
    json_data: dict[str, Any],
    character_pool: set[Character],
    starting_characters: set[Character],
    location_pool: set[Location],
    next_choices: dict[str, SceneChoice],
    player: Character,
    narrator: Character,
    char_by_canonical: dict[str, Character] | None = None,
    loc_by_canonical: dict[str, Location] | None = None,
    char_by_display: dict[str, Character] | None = None,
    loc_by_display: dict[str, Location] | None = None,
) -> list[dict[str, Any]]:
    """Validate a canonical script against scene pools.

    Speakers and location names may use either canonical names or the display
    names declared in the scene TOML.

    :raises RuntimeError: On first validation violation.
    """
    events = json_data.get("events", [])
    if not events:
        raise RuntimeError("Canonical script has no events")

    char_by_canonical = char_by_canonical or {c.canonical_name: c for c in character_pool}
    loc_by_canonical = loc_by_canonical or {l.canonical_name: l for l in location_pool}

    def _char(name: str) -> Character | None:
        return _resolve_by_name(name, char_by_canonical, char_by_display)

    def _loc(name: str) -> Location | None:
        return _resolve_by_name(name, loc_by_canonical, loc_by_display)

    def _char_names() -> list[str]:
        names = set(char_by_canonical)
        if char_by_display:
            names.update(char_by_display)
        return sorted(names)

    def _loc_names() -> list[str]:
        names = set(loc_by_canonical)
        if loc_by_display:
            names.update(loc_by_display)
        return sorted(names)

    # Track here/away state across events for consistency checks.
    # Any character (including player and narrator) may start away.
    here_names: set[str] = {c.canonical_name for c in starting_characters}
    away_names: set[str] = {
        c.canonical_name for c in character_pool
        if c.canonical_name not in here_names
    }

    for idx, ev in enumerate(events):
        event_type = ev.get("event", "")
        speaker = ev.get("speaker", "")

        if event_type not in ("turn", "scene_ended", "story_complete"):
            raise RuntimeError(
                f"Canonical event {idx}: unknown event type '{event_type}'. "
                f"Valid: turn, scene_ended, story_complete"
            )

        # Validate speaker
        if event_type == "turn":
            if _char(speaker) is None:
                raise RuntimeError(
                    f"Canonical event {idx}: speaker '{speaker}' not in character pool. "
                    f"Valid: {_char_names()}"
                )

        # Validate enters (must be away to enter)
        for name in ev.get("enter", []):
            if _char(name) is None:
                raise RuntimeError(
                    f"Canonical event {idx}: enter '{name}' not in character pool. "
                    f"Valid: {_char_names()}"
                )
            canonical = _char(name).canonical_name
            if canonical in here_names:
                raise RuntimeError(
                    f"Canonical event {idx}: character '{name}' is already here."
                )
            if canonical in away_names:
                away_names.discard(canonical)
            here_names.add(canonical)

        # Validate exits (must be here to exit)
        for name in ev.get("exit", []):
            if _char(name) is None:
                raise RuntimeError(
                    f"Canonical event {idx}: exit '{name}' not in character pool. "
                    f"Valid: {_char_names()}"
                )
            canonical = _char(name).canonical_name
            if canonical in away_names:
                raise RuntimeError(
                    f"Canonical event {idx}: character '{name}' is already away."
                )
            if canonical in here_names:
                here_names.discard(canonical)
            away_names.add(canonical)

        # Validate sprite changes
        for char_name, sprite_name in ev.get("sprite_changes", {}).items():
            char = _char(char_name)
            if char is None:
                raise RuntimeError(
                    f"Canonical event {idx}: sprite_change key '{char_name}' not in character pool."
                )
            if sprite_name not in ("none", "hidden") and sprite_name not in char.sprites:
                raise RuntimeError(
                    f"Canonical event {idx}: sprite '{sprite_name}' not available for '{char_name}'. "
                    f"Valid: {char.sprites}"
                )

        # Validate location switch
        switch_loc = ev.get("switch_location")
        if switch_loc and _loc(switch_loc) is None:
            raise RuntimeError(
                f"Canonical event {idx}: switch_location '{switch_loc}' not in location pool. "
                f"Valid: {_loc_names()}"
            )

        # Validate next_scene / choices
        if event_type == "scene_ended":
            choices = ev.get("choices")
            next_scene = ev.get("next_scene")
            if choices:
                if not isinstance(choices, list):
                    raise RuntimeError(
                        f"Canonical event {idx}: 'choices' must be a list."
                    )
                if next_scene:
                    raise RuntimeError(
                        f"Canonical event {idx}: cannot have both 'choices' and 'next_scene'."
                    )
                if idx != len(events) - 1:
                    raise RuntimeError(
                        f"Canonical event {idx}: choices are only allowed on the last event."
                    )
                for c_idx, choice in enumerate(choices):
                    if not isinstance(choice, dict):
                        raise RuntimeError(
                            f"Canonical event {idx} choice {c_idx}: must be a dict."
                        )
                    c_hint = choice.get("hint", "")
                    c_text = choice.get("text", "")
                    c_next = choice.get("next_scene", "")
                    if not c_hint or not isinstance(c_hint, str):
                        raise RuntimeError(
                            f"Canonical event {idx} choice {c_idx}: 'hint' is required and must be a string."
                        )
                    if not c_text or not isinstance(c_text, str):
                        raise RuntimeError(
                            f"Canonical event {idx} choice {c_idx}: 'text' is required and must be a string."
                        )
                    if not c_next or not isinstance(c_next, str):
                        raise RuntimeError(
                            f"Canonical event {idx} choice {c_idx}: 'next_scene' is required and must be a string."
                        )
                    if c_next not in next_choices:
                        raise RuntimeError(
                            f"Canonical event {idx} choice {c_idx}: next_scene '{c_next}' not in next_choices. "
                            f"Valid: {list(next_choices)}"
                        )
            elif next_scene and next_scene not in next_choices:
                raise RuntimeError(
                    f"Canonical event {idx}: next_scene '{next_scene}' not in next_choices. "
                    f"Valid: {list(next_choices)}"
                )

    return events


@dataclass
class SceneChoice:
    """A possible next-scene transition.

    :ivar id: Scene identifier used to resolve the follow-up TOML file.
    :ivar desc: Human-readable description shown to the orchestrator.
    :ivar prereq_scenes: Optional list of prerequisite scene IDs.  When non-empty,
        the choice is only available if the current scene's predecessor is in
        this list.
    """

    id: str
    desc: str
    prereq_scenes: list[str] = field(default_factory=list)
    summarizer_considerations: str = ""


@dataclass
class Scene:
    """Self-contained narrative unit loaded from a TOML file.

    A scene defines its character pool, location pool, plot text, and the
    valid choices for transitioning to subsequent scenes.
    """

    id: str
    language: str
    zeitgeist: str
    tone: str
    scene_type: str
    character_pool: set[Character]
    starting_characters: set[Character]
    player: Character
    narrator: Character
    location_pool: set[Location]
    starting_location: Location
    plot_considerations: str
    plot_story: str
    next_choices: dict[str, SceneChoice]
    canonical_events: list[dict[str, Any]] = field(default_factory=list)
    name: str = ""
    time: str = ""
    world: str = ""
    world_map: str = ""
    asset_story_name: str = ""
    settings: list[str] = field(default_factory=list)
    items: dict[str, Item] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: Path,
        db: ChromaStore,
        config: AraSettings,
        scene_history: list[str] | None = None,
        registry: AssetRegistry | None = None,
        live_characters: dict[str, Character] | None = None,
        live_locations: dict[str, Location] | None = None,
    ) -> Self:
        """Load a scene from a TOML file.

        Characters and locations are resolved through the ephemeral
        :class:`AssetRegistry` rather than by literal directory name.

        :param path: Path to the scene ``.toml`` file.
        :param db: Shared ChromaDB store passed to character loaders.
        :param config: Application settings.
        :param scene_history: List of already-visited scene IDs (for
            ``prereq_scenes`` prerequisite filtering).
        :param registry: Session registry used to resolve characters/locations.
        :param live_characters: Cache of live character objects keyed by
            canonical name.  Used to preserve runtime state across scenes.
        :param live_locations: Cache of live location objects keyed by
            canonical name.
        :return: Initialised :class:`Scene`.
        :raises RuntimeError: If the player, narrator, or starting location
            cannot be found in their respective pools.
        """
        with path.open("rb") as f:
            data = tomllib.load(f)

        language = normalize_language(data.get("language", config.language))

        char_data = data.get("character", {})
        char_pool_names = char_data.get("pool", [])
        char_init_names = set(char_data.get("inits", []))
        char_init_sprites = char_data.get("init_sprites", [])
        char_sprites_table: dict[str, Any] = char_data.get("sprites", {})
        char_titles_table: dict[str, str] = char_data.get("titles", {})
        char_init_titles = char_data.get("titles", []) if not char_titles_table else []
        player_name = char_data.get("player", "")
        narrator_name = char_data.get("narrator", "")
        anonymous_data: dict[str, Any] = data.get("anonymous", {})

        # Scene TOMLs reference characters and locations by canonical ID (the
        # asset directory name).  We keep both canonical and display maps so
        # LLM-facing output can use display names while internal state uses IDs.
        character_pool: set[Character] = set()
        player: Character | None = None
        narrator: Character | None = None
        char_by_canonical: dict[str, Character] = {}
        char_by_display: dict[str, Character] = {}

        live_chars = live_characters if live_characters is not None else {}
        story_name = path.parent.name
        assets_cc = config.characters_path(story_name)
        for canonical_name in char_pool_names:
            char: Character | None = None
            if registry is not None:
                meta = registry.get_character(canonical_name, language)
                if meta is not None:
                    canonical = meta["canonical_name"]
                    asset_dir = Path(meta["asset_dir"])
                    if canonical in live_chars:
                        char = live_chars[canonical]
                    else:
                        char = load_character(asset_dir, db, language)
                        live_chars[canonical] = char
            # Fallback to direct directory lookup for unregistered assets or
            # callers that do not provide a registry. Reuse the live cache if
            # this character was already loaded.
            if char is None:
                if canonical_name in live_chars:
                    char = live_chars[canonical_name]
                else:
                    char_path = assets_cc / canonical_name
                    if char_path.exists():
                        char = load_character(char_path, db, language)
                        live_chars[canonical_name] = char
            if char is None and canonical_name in anonymous_data:
                anon_entry = anonymous_data[canonical_name]
                if isinstance(anon_entry, dict):
                    char = create_anonymous_character(
                        canonical_name,
                        description=anon_entry.get("desc", ""),
                        sprite=anon_entry.get("sprite", ""),
                    )
                else:
                    char = create_anonymous_character(
                        canonical_name, description=str(anon_entry)
                    )
            if char is None:
                raise RuntimeError(
                    f"Character '{canonical_name}' not found in registry "
                    f"and not defined in [anonymous]."
                )
            character_pool.add(char)
            char_by_canonical[char.canonical_name] = char
            char_by_display[char.name] = char
            if canonical_name == player_name:
                player = char
            if canonical_name == narrator_name:
                narrator = char

        if player is None:
            raise RuntimeError(f"Player character '{player_name}' not found in pool.")
        if narrator is None:
            raise RuntimeError(f"Narrator character '{narrator_name}' not found in pool.")

        def _per_char_value(
            table: dict[str, str],
            positional: list[str],
            canonical: str,
            idx: int,
        ) -> str | None:
            if canonical in table:
                return table[canonical]
            # Allow table keys to be display names for backwards compatibility.
            char = char_by_canonical.get(canonical)
            if char is not None and char.name in table:
                return table[char.name]
            if idx < len(positional):
                return positional[idx]
            return None

        # Apply per-scene initial sprites and titles.
        # The special sprite value "none" means the character is present but invisible.
        for idx, canonical_name in enumerate(char_pool_names):
            char = char_by_canonical[canonical_name]
            sprite_entry = _per_char_value(char_sprites_table, char_init_sprites, canonical_name, idx)
            sprite_name = ""
            visible_to: set[str] = set()
            if isinstance(sprite_entry, dict):
                sprite_name = sprite_entry.get("sprite", "") or ""
                visible_to = set(sprite_entry.get("visible_to", []))
            elif sprite_entry is not None:
                sprite_name = sprite_entry

            if sprite_name:
                if sprite_name in ("none", "hidden"):
                    char.current_sprite = "none"
                    char.hidden = sprite_name == "hidden"
                    # visible_to references canonical IDs in scene TOMLs.
                    char.visible_to = visible_to
                else:
                    char.current_sprite = sprite_name
                    char.hidden = False
                    char.visible_to = visible_to

            title = _per_char_value(char_titles_table, char_init_titles, canonical_name, idx)
            if title is not None:
                char.title = title

        starting_characters = {
            char_by_canonical[name] for name in char_init_names
            if name in char_by_canonical
        }

        loc_data = data.get("location", {})
        loc_pool_names = loc_data.get("pool", [])
        loc_init = loc_data.get("init", "")
        loc_descs = loc_data.get("descs", {})
        loc_loadings = loc_data.get("loading", {})

        location_pool: set[Location] = set()
        starting_location: Location | None = None
        loc_by_canonical: dict[str, Location] = {}
        loc_by_display: dict[str, Location] = {}
        live_locs = live_locations if live_locations is not None else {}
        assets_lc = config.locations_path(story_name)
        for canonical_name in loc_pool_names:
            loc: Location | None = None
            if registry is not None:
                meta = registry.get_location(canonical_name, language)
                if meta is not None:
                    canonical = meta["canonical_name"]
                    asset_dir = Path(meta["asset_dir"])
                    if canonical in live_locs:
                        loc = live_locs[canonical]
                    else:
                        loc = load_location(
                            asset_dir,
                            name=canonical,
                            inline_desc=loc_descs.get(canonical_name, ""),
                            inline_loading=loc_loadings.get(canonical_name, ""),
                            language=language,
                        )
                        live_locs[canonical] = loc
            # Fallback to direct directory lookup for locations not in registry.
            # Reuse the live cache if this location was already loaded.
            if loc is None:
                if canonical_name in live_locs:
                    loc = live_locs[canonical_name]
                else:
                    loc_path = assets_lc / canonical_name
                    loc = load_location(
                        loc_path,
                        name=canonical_name,
                        inline_desc=loc_descs.get(canonical_name, ""),
                        inline_loading=loc_loadings.get(canonical_name, ""),
                        language=language,
                    )
                    live_locs[canonical_name] = loc
            location_pool.add(loc)
            loc_by_canonical[loc.canonical_name] = loc
            loc_by_display[loc.name] = loc
            if canonical_name == loc_init:
                starting_location = loc

        if starting_location is None:
            raise RuntimeError(f"Starting location '{loc_init}' not found in pool.")

        world_data = data.get("world", {})
        world_map = world_data.get("map", "") if isinstance(world_data, dict) else ""

        plot_data = data.get("plot", {})
        plot_considerations = plot_data.get("considerations", "")
        plot_story = plot_data.get("scene", "")

        # Load optional plot item templates. Filenames are relative to
        # data/assets/items/<story>/ or data/assets/items/.
        items: dict[str, Item] = {}
        for item_file in plot_data.get("items", []):
            if not isinstance(item_file, str):
                continue
            if not item_file.endswith(".toml"):
                item_file += ".toml"
            item_path = config.items_path(story_name) / item_file
            if not item_path.exists():
                item_path = config.items_path() / item_file
            try:
                item = load_item(item_path)
            except Exception as exc:
                logger.warning(f"Could not load plot item {item_file}: {exc}")
                continue
            items[item.id] = item

        next_raw = plot_data.get("next", {})
        next_raw.pop("considerations", None)
        next_choices: dict[str, SceneChoice] = {}
        for choice_id, details in next_raw.items():
            if isinstance(details, dict):
                next_choices[choice_id] = SceneChoice(
                    id=choice_id,
                    desc=details.get("desc", ""),
                    prereq_scenes=details.get("prereq_scenes", []),
                    summarizer_considerations=details.get(
                        "summarizer_considerations", ""
                    ),
                )
            else:
                next_choices[choice_id] = SceneChoice(id=choice_id, desc=str(details))

        # Filter choices based on prereq_scenes logic.
        # A choice is only available if every scene in its prereq_scenes list has
        # been visited (including the current scene itself).
        scene_id = data.get("id", path.stem)
        scene_name = data.get("name", scene_id)
        visited = set(scene_history or []) | {scene_id}
        has_prereq_scenes = any(c.prereq_scenes for c in next_choices.values())
        if has_prereq_scenes:
            valid = {
                k: v for k, v in next_choices.items()
                if not v.prereq_scenes or all(of in visited for of in v.prereq_scenes)
            }
        else:
            valid = next_choices

        # Parse optional canonical script
        canonical_events: list[dict[str, Any]] = []
        canonical_data = data.get("canonical", {})
        canonical_script = canonical_data.get("script", "")
        if canonical_script:
            script_path = path.parent / canonical_script
            if not script_path.exists():
                raise RuntimeError(f"Canonical script not found: {script_path}")
            with script_path.open("r", encoding="utf-8") as f:
                canonical_json = json.load(f)
            canonical_events = _validate_canonical_events(
                canonical_json,
                character_pool,
                starting_characters,
                location_pool,
                valid,
                player,
                narrator,
                char_by_canonical,
                loc_by_canonical,
                char_by_display,
                loc_by_display,
            )

        logger.debug(f"Loaded scene {scene_name} ({scene_id}) with {len(character_pool)} chars, {len(location_pool)} locs")
        return cls(
            id=scene_id,
            name=scene_name,
            language=language,
            zeitgeist=data.get("zeitgeist", ""),
            tone=data.get("tone", ""),
            time=data.get("time", ""),
            world=data.get("world", ""),
            world_map=world_map,
            settings=list(data.get("settings", [])) if isinstance(data.get("settings"), (list, tuple)) else [],
            asset_story_name=story_name,
            scene_type=data.get("type", "fin" if path.name == "fin_scene.toml" else "normal"),
            character_pool=character_pool,
            starting_characters=starting_characters,
            player=player,
            narrator=narrator,
            location_pool=location_pool,
            starting_location=starting_location,
            plot_considerations=plot_considerations,
            plot_story=plot_story,
            next_choices=valid,
            canonical_events=canonical_events,
            items=items,
        )

    def next_choices_pretty(self) -> str:
        """Return a human-readable list of next-scene choices.

        :return: One ``id: desc`` line per choice.
        """
        return "\n".join(
            f"{c.id}: {c.desc}" for c in self.next_choices.values()
        )

    def plot_as_tool_content(self) -> str:
        """Format the plot data as a single string for injection into the
        orchestrator's system prompt.

        :return: Combined considerations, story, world map, and next-choice list.
        """
        parts = [
            f"Considerations:\n{self.plot_considerations}",
            f"Plot:\n{self.plot_story}",
        ]
        if self.items:
            item_lines = []
            for item in self.items.values():
                meta = f" metadata={item.metadata}" if item.metadata else ""
                item_lines.append(f"  {item.id}: {item.name} - {item.description}{meta}")
            parts.append("Available item templates:\n" + "\n".join(item_lines))
        if self.world_map:
            parts.append(f"World map:\n{self.world_map}")
        parts.append(f"Next scene choices:\n{self.next_choices_pretty()}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Canonical / display name resolution helpers
    # ------------------------------------------------------------------ #
    def character_by_canonical(self, name: str) -> Character | None:
        """Return the character whose canonical ID equals *name*."""
        for char in self.character_pool:
            if char.canonical_name == name:
                return char
        return None

    def character_by_display(self, name: str) -> Character | None:
        """Return the character whose display name equals *name*."""
        for char in self.character_pool:
            if char.name == name:
                return char
        return None

    def character_by_name(self, name: str) -> Character | None:
        """Resolve a character by canonical or display name."""
        return self.character_by_canonical(name) or self.character_by_display(name)

    def location_by_canonical(self, name: str) -> Location | None:
        """Return the location whose canonical ID equals *name*."""
        for loc in self.location_pool:
            if loc.canonical_name == name:
                return loc
        return None

    def location_by_display(self, name: str) -> Location | None:
        """Return the location whose display name equals *name*."""
        for loc in self.location_pool:
            if loc.name == name:
                return loc
        return None

    def location_by_name(self, name: str) -> Location | None:
        """Resolve a location by canonical or display name."""
        return self.location_by_canonical(name) or self.location_by_display(name)

    def display_name_for(self, canonical: str) -> str:
        """Return the display name for *canonical* ID, or the ID itself."""
        char = self.character_by_canonical(canonical)
        if char is not None:
            return char.name
        loc = self.location_by_canonical(canonical)
        if loc is not None:
            return loc.name
        return canonical
