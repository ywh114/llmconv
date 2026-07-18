"""Player system-page DSL.

The system page is a structured player-facing status overlay.  It is stored as a
sectioned DSL so the orchestrator, summarizer, and frontend can all reason about
it consistently.

Inventory items inside an ``inventory`` section may be plain strings or dicts
with optional ``name``, ``description``, and ``metadata`` fields.  ``metadata``
is an arbitrary dict that the engine and frontend can use for plot-relevant
item behavior (e.g. ``{"unlocks": "sealed_door"}``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SystemPage:
    """Player system page rendered by the webclient on the ``E`` key overlay.

    :ivar title: Page title, e.g. ``"Commander Status"``.
    :ivar sections: Ordered list of sections.  Each section has a ``type`` and
        an ``items`` list.  Supported types are ``bars``, ``inventory``,
        ``skills``, and ``text``.
    """

    title: str = "Status"
    sections: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the page to its sectioned DSL representation."""
        return {"title": self.title, "sections": list(self.sections)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SystemPage:
        """Deserialize from a sectioned DSL dict."""
        if not data:
            return cls()
        return cls(
            title=data.get("title", "Status"),
            sections=list(data.get("sections", [])),
        )

    @classmethod
    def from_legacy(cls, legacy: dict[str, Any]) -> SystemPage:
        """Convert the old flat-key ``bars/inventory/skills`` dict to DSL."""
        sections: list[dict[str, Any]] = []
        if not legacy:
            return cls()
        if "bars" in legacy and isinstance(legacy["bars"], dict):
            sections.append({
                "type": "bars",
                "items": [
                    {"label": name, "value": value, "max": 100}
                    for name, value in legacy["bars"].items()
                ],
            })
        for key in ("inventory", "skills"):
            if key in legacy and isinstance(legacy[key], list):
                sections.append({"type": key, "items": list(legacy[key])})
        return cls(title=legacy.get("title", "Status"), sections=sections)

    def to_legacy(self) -> dict[str, Any]:
        """Return a backward-compatible flat-key view for legacy consumers."""
        legacy: dict[str, Any] = {"title": self.title}
        for section in self.sections:
            stype = section.get("type")
            items = section.get("items", [])
            if stype == "bars":
                legacy["bars"] = {
                    item.get("label", ""): item.get("value", 0)
                    for item in items
                    if isinstance(item, dict)
                }
            elif stype in ("inventory", "skills"):
                legacy[stype] = list(items)
        return legacy


def apply_page_update(
    page_dict: dict[str, Any],
    changes: dict[str, Any],
    resolve_item: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Merge a status-page update into *page_dict* and return the new dict.

    A ``changes`` dict with ``sections``/``title`` keys merges section by
    section (keyed on ``type`` or ``type:label``, items merged by label);
    anything else is treated as a legacy flat-key update.

    :param resolve_item: Optional callable that fills inventory item fields
        from plot templates before merging.
    """
    resolve = resolve_item or (lambda it: it)
    page = SystemPage.from_dict(page_dict)
    if "sections" in changes or "title" in changes:
        new_page = SystemPage.from_dict(changes)
        if new_page.title:
            page.title = new_page.title
        existing: dict[str, dict[str, Any]] = {}
        for s in page.sections:
            key = s.get("label")
            if key:
                key = f"{s.get('type')}:{key}"
            else:
                key = s.get("type")
            existing[key] = s
        for section in new_page.sections:
            stype = section.get("type")
            if stype == "inventory":
                section = dict(section)
                section["items"] = [
                    resolve(it) for it in section.get("items", [])
                ]
            key = section.get("label")
            if key:
                key = f"{stype}:{key}"
            else:
                key = stype
            prev = existing.get(key)
            if prev and prev.get("items") and section.get("items"):
                # Merge items by their label, replacing matches
                merged = []
                seen: set[str] = set()

                def _item_label(item: Any) -> str:
                    if isinstance(item, dict):
                        return str(item.get("label", ""))
                    return str(item)

                for item in section.get("items", []):
                    merged.append(item)
                    seen.add(_item_label(item))
                for item in prev.get("items", []):
                    if _item_label(item) not in seen:
                        merged.append(item)
                existing[key] = dict(section, items=merged)
            else:
                existing[key] = section
        page.sections = list(existing.values())
    else:
        legacy = page.to_legacy()
        legacy.update(changes)
        page = SystemPage.from_legacy(legacy)
    return page.to_dict()


def pretty_print(page: dict[str, Any] | None) -> str:
    """Render a system-page DSL dict as human-readable text.

    Used when injecting a character or location status into a prompt so the LLM
    sees a clean description instead of raw JSON.
    """
    if not page:
        return ""
    title = page.get("title", "Status")
    sections = page.get("sections", [])
    if not sections:
        return ""
    lines: list[str] = [f"{title}:"]
    for section in sections:
        stype = section.get("type", "text")
        label = section.get("label")
        items = section.get("items", [])
        if not items:
            continue
        if label:
            lines.append(f"  [{stype}] {label}")
        else:
            lines.append(f"  [{stype}]")
        for item in items:
            if isinstance(item, str):
                lines.append(f"    - {item}")
            elif isinstance(item, dict):
                name = item.get("name", "")
                desc = item.get("description", "")
                if name and desc:
                    lines.append(f"    - {name}: {desc}")
                elif name:
                    lines.append(f"    - {name}")
                elif desc:
                    lines.append(f"    - {desc}")
                else:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"    - {item}")
    return "\n".join(lines)
