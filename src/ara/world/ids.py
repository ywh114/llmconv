"""Deterministic ID helpers for world entities."""

import uuid


def stable_uuid(kind: str, name: str) -> str:
    """Return a deterministic UUID string for a character or location."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ara.{kind}.{name}"))
