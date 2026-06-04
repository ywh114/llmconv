"""Scene definition and TOML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import tomllib

from ara.config import AraSettings
from ara.memory.chroma import ChromaStore
from ara.world.character import Character, create_anonymous_character, load_character
from ara.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Location:
    """A place within a scene.

    :ivar name: Unique location identifier.
    :ivar desc: Human-readable description used in narrator prompts.
    :ivar lore: Optional extended background information.
    """

    name: str
    desc: str
    lore: str = ""

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Location):
            return NotImplemented
        return self.name == other.name

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


@dataclass
class SceneChoice:
    """A possible next-scene transition.

    :ivar id: Scene identifier used to resolve the follow-up TOML file.
    :ivar desc: Human-readable description shown to the orchestrator.
    :ivar only_for: Optional list of prerequisite scene IDs.  When non-empty,
        the choice is only available if the current scene's predecessor is in
        this list.
    """

    id: str
    desc: str
    only_for: list[str] = field(default_factory=list)


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

    @classmethod
    def load(
        cls,
        path: Path,
        db: ChromaStore,
        config: AraSettings,
        prev_id: str = "",
    ) -> Self:
        """Load a scene from a TOML file.

        Character cards are resolved relative to
        ``config.data_dir / "assets" / "cc"``.

        :param path: Path to the scene ``.toml`` file.
        :param db: Shared ChromaDB store passed to character loaders.
        :param config: Application settings.
        :param prev_id: Identifier of the previous scene (for ``only_for``
            prerequisite filtering).
        :return: Initialised :class:`Scene`.
        :raises RuntimeError: If the player, narrator, or starting location
            cannot be found in their respective pools.
        """
        with path.open("rb") as f:
            data = tomllib.load(f)

        char_data = data.get("character", {})
        char_pool_names = char_data.get("pool", [])
        char_init_names = set(char_data.get("inits", []))
        char_init_sprites = char_data.get("init_sprites", [])
        player_name = char_data.get("player", "")
        narrator_name = char_data.get("narrator", "")
        anonymous_data: dict[str, Any] = data.get("anonymous", {})

        character_pool: set[Character] = set()
        player: Character | None = None
        narrator: Character | None = None

        assets_cc = config.data_dir / "assets" / "cc"
        for name in char_pool_names:
            char_path = assets_cc / name
            if char_path.exists():
                char = load_character(char_path, db)
            elif name in anonymous_data:
                anon_entry = anonymous_data[name]
                if isinstance(anon_entry, dict):
                    char = create_anonymous_character(
                        name,
                        description=anon_entry.get("desc", ""),
                        sprite=anon_entry.get("sprite", ""),
                    )
                else:
                    char = create_anonymous_character(name, description=str(anon_entry))
            else:
                raise RuntimeError(
                    f"Character '{name}' not found in assets ({char_path}) "
                    f"and not defined in [anonymous]."
                )
            character_pool.add(char)
            if char.name == player_name:
                player = char
            if char.name == narrator_name:
                narrator = char

        if player is None:
            raise RuntimeError(f"Player character '{player_name}' not found in pool.")
        if narrator is None:
            raise RuntimeError(f"Narrator character '{narrator_name}' not found in pool.")

        # Apply per-scene initial sprites (parallel to pool order).
        for idx, name in enumerate(char_pool_names):
            sprite = char_init_sprites[idx] if idx < len(char_init_sprites) else "default_neutral"
            for char in character_pool:
                if char.name == name:
                    char.current_sprite = sprite
                    break

        starting_characters = {c for c in character_pool if c.name in char_init_names}

        loc_data = data.get("location", {})
        loc_pool_names = loc_data.get("pool", [])
        loc_init = loc_data.get("init", "")
        loc_descs = loc_data.get("descs", {})

        location_pool: set[Location] = set()
        starting_location: Location | None = None
        for name in loc_pool_names:
            desc = loc_descs.get(name, "")
            loc = Location(name=name, desc=desc)
            location_pool.add(loc)
            if name == loc_init:
                starting_location = loc

        if starting_location is None:
            raise RuntimeError(f"Starting location '{loc_init}' not found in pool.")

        plot_data = data.get("plot", {})
        plot_considerations = plot_data.get("considerations", "")
        plot_story = plot_data.get("scene", "")

        next_raw = plot_data.get("next", {})
        next_raw.pop("considerations", None)
        next_choices: dict[str, SceneChoice] = {}
        for choice_id, details in next_raw.items():
            if isinstance(details, dict):
                next_choices[choice_id] = SceneChoice(
                    id=choice_id,
                    desc=details.get("desc", ""),
                    only_for=details.get("only_for", []),
                )
            else:
                next_choices[choice_id] = SceneChoice(id=choice_id, desc=str(details))

        # Filter choices based on only_for logic
        has_only_for = any(c.only_for for c in next_choices.values())
        if has_only_for:
            valid = {
                k: v for k, v in next_choices.items()
                if not v.only_for or not prev_id or prev_id in v.only_for
            }
        else:
            valid = next_choices

        logger.debug(f"Loaded scene {data.get('id', path.stem)} with {len(character_pool)} chars, {len(location_pool)} locs")
        return cls(
            id=data.get("id", path.stem),
            language=data.get("language", config.language),
            zeitgeist=data.get("zeitgeist", ""),
            tone=data.get("tone", ""),
            scene_type=data.get("type", "normal"),
            character_pool=character_pool,
            starting_characters=starting_characters,
            player=player,
            narrator=narrator,
            location_pool=location_pool,
            starting_location=starting_location,
            plot_considerations=plot_considerations,
            plot_story=plot_story,
            next_choices=valid,
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

        :return: Combined considerations, story, and next-choice list.
        """
        return (
            f"Considerations:\n{self.plot_considerations}\n"
            f"Plot:\n{self.plot_story}\n"
            f"Next scene choices:\n{self.next_choices_pretty()}"
        )
