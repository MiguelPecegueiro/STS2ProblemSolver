"""Import and organize Spire Codex cards into a local sectioned card database."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sts2_agent.knowledge import CODEX_BASE_URL, _fetch_json

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
CARDS_ROOT = PROJECT_ROOT / "data" / "cards"
BY_COLOR_DIR = CARDS_ROOT / "by_color"
INDEX_PATH = CARDS_ROOT / "index.json"

# Playable character colors first, then shared pools (user order + Codex extras).
CARD_SECTIONS: tuple[tuple[str, str], ...] = (
    ("ironclad", "Ironclad"),
    ("silent", "Silent"),
    ("defect", "Defect"),
    ("necrobinder", "Necrobinder"),
    ("regent", "Regent"),
    ("colorless", "Colorless"),
    ("token", "Token"),
    ("curse", "Curse"),
    # Codex also tags these; kept as separate sections so nothing is dropped.
    ("status", "Status"),
    ("event", "Event"),
    ("quest", "Quest"),
)

SECTION_KEYS: frozenset[str] = frozenset(k for k, _ in CARD_SECTIONS)
SECTION_LABELS: dict[str, str] = dict(CARD_SECTIONS)

# Map unknown color strings into a section (extend as Codex adds values).
COLOR_ALIASES: dict[str, str] = {
    "the_ironclad": "ironclad",
    "ironclad": "ironclad",
    "the_silent": "silent",
    "silent": "silent",
    "the_defect": "defect",
    "defect": "defect",
    "the_necrobinder": "necrobinder",
    "necrobinder": "necrobinder",
    "the_regent": "regent",
    "regent": "regent",
    "colorless": "colorless",
    "colourless": "colorless",
    "token": "token",
    "tokens": "token",
    "curse": "curse",
    "curses": "curse",
    "status": "status",
    "event": "event",
    "quest": "quest",
}


@dataclass(frozen=True, slots=True)
class ImportReport:
    fetched_at: str
    total_cards: int
    section_counts: dict[str, int]
    unmapped: list[str]
    index_path: Path
    wrote_paths: list[Path]


def normalize_color_key(raw: str | None) -> str | None:
    if not raw:
        return None
    key = str(raw).strip().lower().replace("the ", "").replace("-", "_").replace(" ", "_")
    return COLOR_ALIASES.get(key, key if key in SECTION_KEYS else None)


def classify_card(card: dict[str, Any]) -> str | None:
    """Return section key for a Codex card dict."""
    for field in ("color_key", "color", "character_key", "character"):
        section = normalize_color_key(card.get(field))
        if section:
            return section
    card_id = str(card.get("id") or "").upper()
    if card_id.startswith("CURSE.") or "CURSE" in card_id:
        return "curse"
    if card_id.startswith("TOKEN.") or "TOKEN" in card_id:
        return "token"
    return None


def _sort_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(cards, key=lambda c: (str(c.get("name") or "").lower(), str(c.get("id") or "")))


def fetch_codex_cards(*, timeout: float = 60.0) -> list[dict[str, Any]]:
    url = f"{CODEX_BASE_URL}/api/cards"
    logger.info("Fetching %s", url)
    data = _fetch_json(url, timeout=timeout)
    if not isinstance(data, list):
        raise ValueError(f"Expected list from Codex cards API, got {type(data).__name__}")
    return [c for c in data if isinstance(c, dict)]


def partition_cards(cards: list[dict[str, Any]]) -> tuple[dict[str, list[dict]], list[str]]:
    buckets: dict[str, list[dict]] = {key: [] for key, _ in CARD_SECTIONS}
    unmapped: list[str] = []
    for card in cards:
        section = classify_card(card)
        name = str(card.get("name") or card.get("id") or "?")
        if section is None:
            raw = card.get("color") or card.get("color_key") or "(missing)"
            unmapped.append(f"{name} (color={raw!r})")
            continue
        enriched = dict(card)
        enriched["section"] = section
        enriched["section_label"] = SECTION_LABELS[section]
        buckets[section].append(enriched)
    for key in buckets:
        buckets[key] = _sort_cards(buckets[key])
    return buckets, unmapped


def write_card_database(
    cards: list[dict[str, Any]],
    *,
    fetched_at: str | None = None,
    root: Path | None = None,
) -> ImportReport:
    """Write index.json and one JSON file per color section."""
    root = root or CARDS_ROOT
    by_color = root / "by_color"
    by_color.mkdir(parents=True, exist_ok=True)

    ts = fetched_at or datetime.now(timezone.utc).isoformat()
    buckets, unmapped = partition_cards(cards)
    wrote: list[Path] = []

    section_meta: dict[str, dict[str, Any]] = {}
    for section_key, label in CARD_SECTIONS:
        section_cards = buckets[section_key]
        rel_path = f"by_color/{section_key}.json"
        out_path = root / rel_path
        payload = {
            "section": section_key,
            "label": label,
            "fetched_at": ts,
            "source": f"{CODEX_BASE_URL}/api/cards",
            "count": len(section_cards),
            "cards": section_cards,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        wrote.append(out_path)
        section_meta[section_key] = {
            "label": label,
            "path": rel_path,
            "count": len(section_cards),
        }

    index = {
        "version": 1,
        "fetched_at": ts,
        "source": f"{CODEX_BASE_URL}/api/cards",
        "total_cards": len(cards),
        "mapped_cards": sum(section_meta[s]["count"] for s in section_meta),
        "unmapped_count": len(unmapped),
        "unmapped": unmapped[:50],
        "sections": section_meta,
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    wrote.insert(0, index_path)

    return ImportReport(
        fetched_at=ts,
        total_cards=len(cards),
        section_counts={k: section_meta[k]["count"] for k in section_meta},
        unmapped=unmapped,
        index_path=index_path,
        wrote_paths=wrote,
    )


def import_codex_cards(*, force: bool = False, root: Path | None = None) -> ImportReport:
    """Fetch from Spire Codex and refresh local `data/cards/` layout."""
    root = root or CARDS_ROOT
    if not force and (root / "index.json").exists():
        try:
            existing = json.loads((root / "index.json").read_text(encoding="utf-8"))
            age = time.time() - datetime.fromisoformat(
                str(existing.get("fetched_at", "")).replace("Z", "+00:00")
            ).timestamp()
            if age < 24 * 60 * 60:
                logger.info("Card database fresh (%.0fh old): %s", age / 3600, root)
                return ImportReport(
                    fetched_at=str(existing.get("fetched_at")),
                    total_cards=int(existing.get("total_cards") or 0),
                    section_counts={
                        k: int(v.get("count") or 0)
                        for k, v in (existing.get("sections") or {}).items()
                    },
                    unmapped=list(existing.get("unmapped") or []),
                    index_path=root / "index.json",
                    wrote_paths=[],
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    cards = fetch_codex_cards()
    return write_card_database(cards, root=root)


def load_section(section: str, *, root: Path | None = None) -> list[dict[str, Any]]:
    root = root or CARDS_ROOT
    path = root / "by_color" / f"{section.lower()}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("cards") or [])


def load_all_cards(*, root: Path | None = None) -> list[dict[str, Any]]:
    root = root or CARDS_ROOT
    index_path = root / "index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    cards: list[dict[str, Any]] = []
    for section in index.get("sections") or {}:
        cards.extend(load_section(section, root=root))
    return cards


def load_card_index(*, root: Path | None = None) -> dict[str, Any]:
    root = root or CARDS_ROOT
    path = root / "index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
