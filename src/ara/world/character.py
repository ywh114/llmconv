"""Character dataclass and PNG card loader."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from enum import IntEnum

from ara.llm.context import ConversationContext
from ara.memory.chroma import ChromaStore
from ara.memory.knowledge import CharacterMemory, NullMemory, Scratchpad
from ara.llm.models import Context
from ara.utils.logger import get_logger
from ara.world.ids import stable_uuid as _stable_uuid_str
from ara.world.system_page import pretty_print

logger = get_logger(__name__)


class Importance(IntEnum):
    """Character importance levels controlling memory fidelity and scratch access.

    Higher importance characters receive end-of-scene scratch updates and
    persistent conversation storage.
    """

    ANONYMOUS = 0
    """Background character with minimal persistence."""

    IMPORTANT = 1
    """Significant character with scratch updates and memory storage."""

    REQUIRED = 2
    """Plot-critical character."""

    EIGEN = 3
    """Protagonist-level character with highest persistence."""


def _stable_uuid(name: str) -> uuid.UUID:
    """Generate a deterministic UUID for *name*."""
    return uuid.UUID(_stable_uuid_str("character", name))


@dataclass
class Character:
    """In-world character backed by a card and persistent memory.

    :ivar id: Stable UUID derived from the character's canonical name.
    :ivar canonical_name: Stable latin identifier (usually the asset directory name).
    :ivar name: Localized display name.
    :ivar card_fields: Key-value pairs loaded from the PNG card metadata.
    :ivar importance: Persistence level controlling scratch updates and
        memory storage.
    :ivar memory: Vector-backed conversation memory.
    :ivar scratch: Ephemeral scratchpad for in-scene reasoning.
    :ivar sprites: List of available sprite names (filename stems) for this
        character, e.g. ``["default_neutral", "default_happy"]``.
    :ivar current_sprite: The sprite currently being displayed.
    :ivar crops: Per-sprite crop regions keyed by sprite name.
    :ivar sprite_descriptions: Base description of each sprite/skin.
    """

    id: uuid.UUID
    canonical_name: str
    name: str
    card_fields: dict[str, str]
    importance: Importance
    memory: CharacterMemory | NullMemory
    scratch: Scratchpad
    sprites: list[str] = field(default_factory=list)
    current_sprite: str = ""
    crops: dict[str, dict[str, Any]] = field(default_factory=dict)
    sprite_descriptions: dict[str, str] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    """System-page DSL dict for the character's persistent status."""
    names: dict[str, str] = field(default_factory=dict)
    """Alternative language names for the character, e.g. {\"zh\": \"柴郡\"}."""
    title: str = ""
    """Persistent title or epithet (e.g. \"Heavy Cruiser\", \"Wizard of Despair\")."""
    hidden: bool = False
    """True when the character is present but hidden from most observers."""
    visible_to: set[str] = field(default_factory=set)
    """Canonical names of characters who can still see/hear this hidden character."""
    inner_log: list[dict[str, Any]] = field(default_factory=list)
    """Private inner thoughts generated for this character."""
    prev_scene_summary: str = ""
    """Tailored bridging summary injected at the start of a new scene."""
    card_overrides: dict[str, str] = field(default_factory=dict)
    """Per-scene overrides for card fields, applied by the transition summarizer."""

    def skin_description(self, sprite: str | None = None) -> str:
        """Return the base description for a sprite, or the current sprite.

        The description is not absolute: the orchestrator or character may
        apply ephemeral modifiers (e.g. muddy dress) via scratch or directives.
        """
        sprite = sprite or self.current_sprite
        return self.sprite_descriptions.get(sprite, "")

    def display_name(self, language: str | None = None) -> str:
        """Return the name to use for *language*.

        Falls back to the canonical English name when no alias exists.
        """
        if language and language != "en":
            return self.names.get(language, self.name)
        return self.name

    def display_name_with_title(self, language: str | None = None) -> str:
        """Return the display name with title prefix if present.

        Returns "[title] name" format when title exists, otherwise just name.
        """
        name = self.display_name(language)
        if self.title:
            return f"[{self.title}] {name}"
        return name

    @property
    def whoami(self) -> Context:
        """Return injected context that establishes the character's identity.

        The context follows a question-and-answer format where the model is
        prompted for each card field and supplies the stored value.
        """
        name = self.name
        fields = [
            ("name", name),
            ("summary", self.card_overrides.get("summary", self.card_fields.get("summary", ""))),
            ("personality", self.card_overrides.get("personality", self.card_fields.get("personality", ""))),
            ("scenario", self.card_overrides.get("scenario", self.card_fields.get("scenario", ""))),
            ("greeting_message", self.card_overrides.get("greeting_message", self.card_fields.get("greeting_message", ""))),
            ("example_messages", self.card_overrides.get("example_messages", self.card_fields.get("example_messages", ""))),
        ]
        ctx: Context = []
        for field_name, value in fields:
            ctx.append({
                "role": "user",
                "content": f"Please provide your `{field_name}`.",
                "name": ConversationContext.default_sysname,
            })
            ctx.append({
                "role": "assistant",
                "content": value,
                "name": name,
            })
        return ctx

    @property
    def status_context(self) -> Context:
        """Return a context snippet that injects the character's persistent status.

        The status is stored as a system-page DSL dict and is pretty-printed
        before being shown to the character.  If no status is set, an empty list
        is returned so the character is not prompted for a status block.
        """
        if not self.status:
            return []
        status_text = pretty_print(self.status)
        if not status_text:
            return []
        return [
            {
                "role": "user",
                "content": "Please provide your `status`.",
                "name": ConversationContext.default_sysname,
            },
            {
                "role": "assistant",
                "content": status_text,
                "name": self.name,
            },
        ]

    @property
    def scene_summary_context(self) -> Context:
        """Return a context snippet that injects the per-character scene summary.

        This is generated by the Summarizer during scene transitions and
        tells the character what they should remember from the previous scene.
        """
        if not self.prev_scene_summary:
            return []
        return [
            {
                "role": "user",
                "content": "Please summarize what happened before this scene.",
                "name": ConversationContext.default_sysname,
            },
            {
                "role": "assistant",
                "content": self.prev_scene_summary,
                "name": self.name,
            },
        ]

    @property
    def scratch_context(self) -> Context:
        """Return a context snippet that injects the character's scratchpad.

        If the scratch is still at its default, the previous scene's scratch
        is shown instead so the character retains continuity.
        """
        empty = self.scratch.text == "Nothing yet!"
        if not empty:
            user_content = (
                "Please provide your `scratch`.\n"
                "Never show this to others, or mention that it exists!"
            )
            assistant_content = self.scratch.text
        else:
            user_content = (
                "Please provide the summary of your `scratch` "
                "from your previous conversation.\n"
                "Never show this to others, or mention that it exists!"
            )
            assistant_content = self.scratch.prev_text

        return [
            {
                "role": "user",
                "content": user_content,
                "name": ConversationContext.default_sysname,
            },
            {
                "role": "assistant",
                "content": assistant_content,
                "name": self.name,
            },
        ]

    def __hash__(self) -> int:
        """Hash based on the character's stable ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality is identity-by-ID."""
        if not isinstance(other, Character):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        """Short representation for debugging."""
        return f"<Character {self.name}>"


def _read_meta(path: Path) -> dict[str, str]:
    """Read an optional ``meta.toml`` sidecar from *path*.

    :param path: Directory that may contain ``meta.toml``.
    :return: Parsed TOML dict or an empty dict if the file does not exist.
    """
    meta_path = path / "meta.toml"
    if meta_path.exists():
        with meta_path.open("rb") as f:
            return tomllib.load(f)
    return {}


def _read_card_toml(path: Path) -> dict[str, str] | None:
    """Read an optional ``card.toml`` sidecar from *path*.

    :param path: Character asset directory.
    :return: Parsed card fields, or ``None`` if the file does not exist.
    """
    card_path = path / "card.toml"
    if card_path.exists():
        with card_path.open("rb") as f:
            data = tomllib.load(f)
        return {
            "name": data.get("name", ""),
            "summary": data.get("summary", ""),
            "personality": data.get("personality", ""),
            "scenario": data.get("scenario", ""),
            "greeting_message": data.get("greeting_message", ""),
            "example_messages": data.get("example_messages", ""),
        }
    return None


def create_anonymous_character(
    name: str,
    description: str = "",
    sprite: str = "",
    title: str = "",
) -> Character:
    """Create a background extra with no assets and no persistent memory.

    :param name: Display name for the background character.
    :param description: Unstructured description of what the character is and
        should do. Stored in ``card_fields["summary"]`` so it appears in the
        character's ``whoami`` context.
    :param sprite: Sprite identifier for the web VN frontend (e.g.
        ``cat_neutral``). Resolved under ``data/assets/cc/<story>/anonymous/``
        if it exists, otherwise ``data/assets/cc/anonymous/``.
    :param title: Optional title or epithet (e.g., "The Unbroken", "Nuclear Fleet Carrier").
        Displays as "[title] name" format.
    :return: A :class:`Character` with :attr:`Importance.ANONYMOUS`.
    """
    char_id = _stable_uuid(name)
    return Character(
        id=char_id,
        canonical_name=name,
        name=name,
        card_fields={
            "name": name,
            "summary": description,
            "personality": "",
            "scenario": "",
            "greeting_message": "",
            "example_messages": "",
            "sprite": sprite,
        },
        importance=Importance.ANONYMOUS,
        memory=NullMemory(),
        scratch=Scratchpad(),
        sprites=[],
        current_sprite="",
        title=title,
    )


def load_character(
    path: Path,
    db: ChromaStore,
    language: str = "en",
) -> Character:
    """Load a character from a directory containing a ``card.toml`` sidecar.

    The directory may also contain PNG sprites and a ``meta.toml`` file with
    an ``importance`` key (one of ``ANONYMOUS``, ``IMPORTANT``, ``REQUIRED``
    or ``EIGEN``).

    :param path: Character asset directory.
    :param db: Shared ChromaDB store.
    :param language: Preferred language for the display name.
    :return: Fully initialised :class:`Character`.
    :raises RuntimeError: If no ``card.toml`` is found.
    """
    pngs = list(path.glob("*.png"))
    card_toml = _read_card_toml(path)

    if card_toml is None:
        raise RuntimeError(f"No card.toml found in {path}")

    card_fields = card_toml

    # The directory name is the canonical ID.  The card ``name`` field and
    # ``[names]`` table provide localized display names.
    canonical = path.name
    display = card_toml.get("name", "") or canonical

    names: dict[str, str] = {}
    card_path = path / "card.toml"
    if card_path.exists():
        with card_path.open("rb") as f:
            raw_card = tomllib.load(f)
        raw_names = raw_card.get("names", {})
        if isinstance(raw_names, dict):
            names = {str(k): str(v) for k, v in raw_names.items() if isinstance(v, str)}
    if canonical and "en" not in names:
        names["en"] = canonical
    if display and language not in names:
        names[language] = display
    name = names.get(language, names.get("en", canonical))

    meta = _read_meta(path)
    importance_name = meta.get("importance", "ANONYMOUS")
    importance = Importance.__members__.get(importance_name, Importance.ANONYMOUS)

    # Parse per-sprite crop regions, focus points, and descriptions from meta.toml
    # (e.g. [default_neutral] topleft = [0,0])
    crops: dict[str, dict[str, Any]] = {}
    sprite_descriptions: dict[str, str] = {}
    for key, value in meta.items():
        if key in ("importance", "sprites", "default_sprite"):
            continue
        if not isinstance(value, dict):
            continue
        crop: dict[str, Any] = {}
        if "topleft" in value and "bottomright" in value:
            crop["topleft"] = value["topleft"]
            crop["bottomright"] = value["bottomright"]
        if "center" in value:
            crop["center"] = value["center"]
        if "focus" in value:
            crop["focus"] = value["focus"]
        if crop:
            crops[key] = crop
        if "description" in value and isinstance(value["description"], str):
            sprite_descriptions[key] = value["description"]

    char_id = _stable_uuid(canonical)
    title = meta.get("title", "")

    # Ensure a display name exists. If the card has no name field and no
    # [names] table, fall back to the canonical id so internal code always
    # has a usable display label.
    if not name:
        name = canonical

    # Sprite list: explicit meta.toml ``sprites`` array, or discover PNGs.
    sprites = meta.get("sprites")
    if not sprites:
        sprites = [f.stem for f in path.glob("*.png")]
    current_sprite = meta.get("default_sprite", sprites[0] if sprites else "")

    logger.debug(f"Loaded character {name} (importance={importance_name})")
    return Character(
        id=char_id,
        canonical_name=canonical,
        name=name,
        card_fields=card_fields,
        importance=importance,
        memory=CharacterMemory(character_id=char_id, db=db),
        scratch=Scratchpad(),
        sprites=sprites,
        current_sprite=current_sprite,
        crops=crops,
        sprite_descriptions=sprite_descriptions,
        names=names,
        title=title,
    )
