"""Language-code normalization for scenes and asset cards."""

from __future__ import annotations

# Map common free-form language strings to short codes matching the [names]
# table keys in card.toml (e.g. "en", "zh").
LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en",
    "en": "en",
    "中文": "zh",
    "chinese": "zh",
    "zh": "zh",
}


def normalize_language(value: str) -> str:
    """Return a canonical short language code for *value*.

    Unknown values are returned unchanged so the caller can still use custom
    language tags.
    """
    return LANGUAGE_ALIASES.get(value.strip().lower(), value)
