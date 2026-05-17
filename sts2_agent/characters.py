"""Canonical character display names (agent + human imports)."""

from __future__ import annotations

CANONICAL_CHARACTERS: dict[str, str] = {
    "IRONCLAD": "The Ironclad",
    "SILENT": "The Silent",
    "DEFECT": "The Defect",
    "NECROBINDER": "The Necrobinder",
    "REGENT": "The Regent",
}


def normalize_character_name(value: object) -> str:
    """Map API / import variants to one display name per character."""
    if value is None:
        return "Unknown"
    text = str(value).strip()
    if not text:
        return "Unknown"

    upper = text.upper()
    if upper.startswith("CHARACTER."):
        upper = upper.split(".", 1)[-1]
    if upper.startswith("THE "):
        upper = upper[4:].strip()

    compact = upper.replace(" ", "_")
    for key, label in CANONICAL_CHARACTERS.items():
        if compact == key or key in compact:
            return label

    return text
