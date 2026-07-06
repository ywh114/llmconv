"""First-class plot-relevant Item objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ara.config import AraSettings

try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


@dataclass
class Item:
    """A plot-important inventory item.

    Mundane inventory strings remain in the system-page inventory section;
    Items carry structured data for plot-critical objects.
    """

    id: str
    name: str
    description: str = ''
    icon: str = ''
    quantity: int = 1
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'quantity': self.quantity,
            'tags': list(self.tags),
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Item:
        """Restore from a serialized dict."""
        return cls(
            id=str(data['id']),
            name=str(data.get('name', data['id'])),
            description=str(data.get('description', '')),
            icon=str(data.get('icon', '')),
            quantity=int(data.get('quantity', 1)),
            tags=list(data.get('tags', [])),
            metadata=dict(data.get('metadata', {})),
        )


def load_item(path: Path) -> Item:
    """Load an item card from a TOML file.

    :param path: Path to the ``.toml`` card.
    :return: Parsed :class:`Item`.
    :raises ValueError: If the card is missing ``id`` or malformed.
    """
    with path.open('rb') as f:
        data = tomllib.load(f)

    item_id = data.get('id')
    if not item_id:
        raise ValueError(f"Item card {path} is missing 'id'.")

    return Item(
        id=str(item_id),
        name=str(data.get('name', item_id)),
        description=str(data.get('description', '')),
        icon=str(data.get('icon', '')),
        quantity=int(data.get('quantity', 1)),
        tags=list(data.get('tags', [])),
        metadata=dict(data.get('metadata', {})),
    )


def load_item_by_id(data_dir: Path, item_id: str, story: str | None = None) -> Item | None:
    """Lazy-load an item card from ``data/assets/items/<id>.toml``.

    Per-story items take priority over global items.

    :param data_dir: Project data directory.
    :param item_id: Item identifier.
    :param story: Optional story id for per-story lookup.
    :return: Parsed :class:`Item` or ``None`` if no card exists.
    """
    config = AraSettings(data_dir=data_dir)
    if story:
        path = config.items_path(story) / f'{item_id}.toml'
        if path.exists():
            return load_item(path)
    path = config.items_path() / f'{item_id}.toml'
    if not path.exists():
        return None
    return load_item(path)
