"""Import and organize Spire Codex monsters into a local sectioned database."""

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
MONSTERS_ROOT = PROJECT_ROOT / "data" / "monsters"
BY_TYPE_DIR = MONSTERS_ROOT / "by_type"

MONSTER_SECTIONS: tuple[tuple[str, str], ...] = (
    ("normal", "Normal"),
    ("elite", "Elite"),
    ("boss", "Boss"),
    ("other", "Other"),
)

SECTION_KEYS: frozenset[str] = frozenset(k for k, _ in MONSTER_SECTIONS)
SECTION_LABELS: dict[str, str] = dict(MONSTER_SECTIONS)

TYPE_ALIASES: dict[str, str] = {
    "normal": "normal",
    "elite": "elite",
    "boss": "boss",
}


@dataclass(frozen=True, slots=True)
class ImportReport:
    fetched_at: str
    total_monsters: int
    section_counts: dict[str, int]
    unmapped: list[str]
    index_path: Path
    wrote_paths: list[Path]


def normalize_type_key(raw: str | None) -> str | None:
    if not raw:
        return None
    key = str(raw).strip().lower()
    return TYPE_ALIASES.get(key, key if key in SECTION_KEYS else None)


def primary_act(monster: dict[str, Any]) -> str | None:
    for enc in monster.get("encounters") or []:
        if not isinstance(enc, dict):
            continue
        act = enc.get("act")
        if act:
            return str(act)
    return None


def classify_monster(monster: dict[str, Any]) -> str | None:
    section = normalize_type_key(monster.get("type"))
    if section:
        return section
    for enc in monster.get("encounters") or []:
        if not isinstance(enc, dict):
            continue
        room = str(enc.get("room_type") or "").lower()
        if room in TYPE_ALIASES:
            return TYPE_ALIASES[room]
    return None


def _sort_monsters(monsters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(monsters, key=lambda m: (str(m.get("name") or "").lower(), str(m.get("id") or "")))


def fetch_codex_monsters(*, timeout: float = 60.0) -> list[dict[str, Any]]:
    url = f"{CODEX_BASE_URL}/api/monsters"
    logger.info("Fetching %s", url)
    data = _fetch_json(url, timeout=timeout)
    if not isinstance(data, list):
        raise ValueError(f"Expected list from Codex monsters API, got {type(data).__name__}")
    return [m for m in data if isinstance(m, dict)]


def partition_monsters(
    monsters: list[dict[str, Any]],
) -> tuple[dict[str, list[dict]], list[str]]:
    buckets: dict[str, list[dict]] = {key: [] for key, _ in MONSTER_SECTIONS}
    unmapped: list[str] = []
    for monster in monsters:
        section = classify_monster(monster)
        name = str(monster.get("name") or monster.get("id") or "?")
        if section is None:
            raw = monster.get("type") or "(missing)"
            unmapped.append(f"{name} (type={raw!r})")
            continue
        enriched = dict(monster)
        enriched["section"] = section
        enriched["section_label"] = SECTION_LABELS[section]
        enriched["primary_act"] = primary_act(monster)
        buckets[section].append(enriched)
    for key in buckets:
        buckets[key] = _sort_monsters(buckets[key])
    return buckets, unmapped


def write_monster_database(
    monsters: list[dict[str, Any]],
    *,
    fetched_at: str | None = None,
    root: Path | None = None,
) -> ImportReport:
    root = root or MONSTERS_ROOT
    by_type = root / "by_type"
    by_type.mkdir(parents=True, exist_ok=True)

    ts = fetched_at or datetime.now(timezone.utc).isoformat()
    buckets, unmapped = partition_monsters(monsters)
    wrote: list[Path] = []

    section_meta: dict[str, dict[str, Any]] = {}
    for section_key, label in MONSTER_SECTIONS:
        section_monsters = buckets[section_key]
        rel_path = f"by_type/{section_key}.json"
        out_path = root / rel_path
        payload = {
            "section": section_key,
            "label": label,
            "fetched_at": ts,
            "source": f"{CODEX_BASE_URL}/api/monsters",
            "count": len(section_monsters),
            "monsters": section_monsters,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        wrote.append(out_path)
        section_meta[section_key] = {
            "label": label,
            "path": rel_path,
            "count": len(section_monsters),
        }

    index = {
        "version": 1,
        "fetched_at": ts,
        "source": f"{CODEX_BASE_URL}/api/monsters",
        "total_monsters": len(monsters),
        "mapped_monsters": sum(section_meta[s]["count"] for s in section_meta),
        "unmapped_count": len(unmapped),
        "unmapped": unmapped[:50],
        "sections": section_meta,
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    wrote.insert(0, index_path)

    return ImportReport(
        fetched_at=ts,
        total_monsters=len(monsters),
        section_counts={k: section_meta[k]["count"] for k in section_meta},
        unmapped=unmapped,
        index_path=index_path,
        wrote_paths=wrote,
    )


def import_codex_monsters(*, force: bool = False, root: Path | None = None) -> ImportReport:
    """Fetch from Spire Codex and refresh local `data/monsters/` layout."""
    root = root or MONSTERS_ROOT
    if not force and (root / "index.json").exists():
        try:
            existing = json.loads((root / "index.json").read_text(encoding="utf-8"))
            age = time.time() - datetime.fromisoformat(
                str(existing.get("fetched_at", "")).replace("Z", "+00:00")
            ).timestamp()
            if age < 24 * 60 * 60:
                logger.info("Monster database fresh (%.0fh old): %s", age / 3600, root)
                return ImportReport(
                    fetched_at=str(existing.get("fetched_at")),
                    total_monsters=int(existing.get("total_monsters") or 0),
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

    monsters = fetch_codex_monsters()
    return write_monster_database(monsters, root=root)


def load_section(section: str, *, root: Path | None = None) -> list[dict[str, Any]]:
    root = root or MONSTERS_ROOT
    path = root / "by_type" / f"{section.lower()}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("monsters") or [])


def load_all_monsters(*, root: Path | None = None) -> list[dict[str, Any]]:
    root = root or MONSTERS_ROOT
    index_path = root / "index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text(encoding="utf-8"))
    monsters: list[dict[str, Any]] = []
    for section in index.get("sections") or {}:
        monsters.extend(load_section(section, root=root))
    return monsters


def load_monster_index(*, root: Path | None = None) -> dict[str, Any]:
    root = root or MONSTERS_ROOT
    path = root / "index.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
