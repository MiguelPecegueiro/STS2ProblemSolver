#!/usr/bin/env python3
"""Scrape Mobalytics STS2 tier lists and character guides into cache/expert_knowledge.json."""

from __future__ import annotations

import argparse
import html as html_lib
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.knowledge import KNOWLEDGE_CACHE_PATH  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_PATH = CACHE_DIR / "expert_knowledge.json"

USER_AGENT = "STS2ProblemSolver/1.0 (local agent; knowledge scraper)"
REQUEST_HEADERS = {"User-Agent": USER_AGENT}

TIER_LIST_URLS = {
    "cards": "https://mobalytics.gg/slay-the-spire-2/tier-lists/cards",
    "relics": "https://mobalytics.gg/slay-the-spire-2/tier-lists/relics",
    "potions": "https://mobalytics.gg/slay-the-spire-2/tier-lists/potions",
}

CHARACTER_GUIDE_URLS = {
    "ironclad": "https://mobalytics.gg/slay-the-spire-2/characters/ironclad-guide",
    "silent": "https://mobalytics.gg/slay-the-spire-2/characters/silent-guide",
    "defect": "https://mobalytics.gg/slay-the-spire-2/characters/defect-guide",
    "regent": "https://mobalytics.gg/slay-the-spire-2/characters/regent-guide",
    "necrobinder": "https://mobalytics.gg/slay-the-spire-2/characters/necrobinder-guide",
}

STANDARD_TIERS = frozenset({"S", "A", "B", "C", "D"})

# Map Mobalytics potion labels to letter tiers for scoring.
POTION_TIER_LETTER = {
    "good always": "S",
    "good early": "A",
    "good late": "A",
    "niche/avg.": "B",
    "niche": "B",
    "mediocre": "D",
}

TIER_RANK = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}

BUILD_ARCHETYPE_MAP = {
    "strength": "strength scaling",
    "block": "block body slam",
    "exhaust": "exhaust synergy",
    "bloodletting": "self-damage bloodletting",
    "strike": "strike deck",
}

logger = logging.getLogger(__name__)


def slug_to_id(slug: str) -> str:
    return slug.strip().upper().replace("-", "_").replace(" ", "_")


def normalize_tier_label(name: str) -> str:
    text = str(name or "").strip()
    if text.upper() in STANDARD_TIERS:
        return text.upper()
    mapped = POTION_TIER_LETTER.get(text.lower())
    if mapped:
        return mapped
    return text


def tier_rank(tier: str) -> int:
    letter = normalize_tier_label(tier)
    if letter in TIER_RANK:
        return TIER_RANK[letter]
    return 0


def fetch_page(url: str, *, timeout: float = 45.0) -> str:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def extract_json_object_after_key(html: str, key: str) -> dict | None:
    pattern = rf'"{re.escape(key)}"\s*:\s*\{{'
    match = re.search(pattern, html)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    for index in range(start, len(html)):
        char = html[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start : index + 1])
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse JSON for %s: %s", key, exc)
                    return None
    return None


def _index_codex_entries(entries: list[dict], bucket: dict[str, dict]) -> None:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "")
        eid = str(entry.get("id") or "").upper()
        if name:
            slug = name.lower().replace(" ", "-")
            bucket[slug] = entry
            bucket[name.lower()] = entry
        if eid:
            bucket[eid.lower()] = entry


def load_codex_indexes() -> tuple[dict[str, dict], dict[str, str]]:
    """slug/name -> codex entry; card name -> character color."""
    if not KNOWLEDGE_CACHE_PATH.exists():
        logger.warning("Codex cache missing at %s - run agent once to populate", KNOWLEDGE_CACHE_PATH)
        return {}, {}
    raw = json.loads(KNOWLEDGE_CACHE_PATH.read_text(encoding="utf-8"))
    by_slug: dict[str, dict] = {}
    char_by_name: dict[str, str] = {}
    for card in raw.get("cards") or []:
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or "")
        if name:
            char_by_name[name.lower()] = str(card.get("color") or "").lower()
    _index_codex_entries(raw.get("cards") or [], by_slug)
    _index_codex_entries(raw.get("relics") or [], by_slug)
    _index_codex_entries(raw.get("potions") or [], by_slug)
    return by_slug, char_by_name


def infer_list_character(
    items: list[dict],
    *,
    char_by_name: dict[str, str],
) -> str | None:
    counts: Counter[str] = Counter()
    for item in items:
        name = str(item.get("name") or "")
        color = char_by_name.get(name.lower())
        if color:
            counts[color] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def resolve_item_id(item: dict, *, codex_by_slug: dict[str, dict]) -> str | None:
    slug = str(item.get("slug") or "").strip()
    name = str(item.get("name") or "").strip()
    if slug and slug.lower() in codex_by_slug:
        codex = codex_by_slug[slug.lower()]
        return str(codex.get("id") or slug_to_id(slug))
    if name:
        slug_guess = name.lower().replace(" ", "-")
        if slug_guess in codex_by_slug:
            codex = codex_by_slug[slug_guess]
            return str(codex.get("id") or slug_to_id(slug_guess))
        return slug_to_id(slug_guess)
    if slug:
        return slug_to_id(slug)
    return None


def merge_tier_entry(
    bucket: dict[str, dict],
    item_id: str,
    tier: str,
    notes: str,
) -> None:
    tier_letter = normalize_tier_label(tier)
    existing = bucket.get(item_id)
    if not existing:
        bucket[item_id] = {"tier": tier_letter, "notes": notes}
        return
    if tier_rank(tier_letter) > tier_rank(str(existing.get("tier") or "")):
        existing["tier"] = tier_letter
    prev = str(existing.get("notes") or "")
    if notes and notes not in prev:
        existing["notes"] = f"{prev}; {notes}" if prev else notes


def parse_tier_lists(
    html: str,
    *,
    codex_by_slug: dict[str, dict],
    char_by_name: dict[str, str],
) -> dict[str, dict]:
    data = extract_json_object_after_key(html, "tierLists")
    if not data:
        return {}

    out: dict[str, dict] = {}
    for tier_list in data.get("values") or []:
        if not isinstance(tier_list, dict):
            continue
        all_items: list[dict] = []
        for section in tier_list.get("tierSections") or []:
            if not isinstance(section, dict):
                continue
            for item in section.get("staticDataItems") or []:
                if isinstance(item, dict):
                    all_items.append(item)

        character = infer_list_character(all_items, char_by_name=char_by_name)

        for section in tier_list.get("tierSections") or []:
            if not isinstance(section, dict):
                continue
            tier_name = str(section.get("name") or "").strip()
            section_desc = str(section.get("description") or "").strip()
            for item in section.get("staticDataItems") or []:
                if not isinstance(item, dict):
                    continue
                item_id = resolve_item_id(item, codex_by_slug=codex_by_slug)
                if not item_id:
                    continue
                if character:
                    notes = f"Mobalytics {character} tier {tier_name}"
                else:
                    notes = f"Mobalytics tier {tier_name}"
                if section_desc:
                    notes = f"{notes} - {section_desc}"
                merge_tier_entry(out, item_id, tier_name, notes)

    return out


def parse_relic_or_potion_page(
    html: str,
    *,
    codex_by_slug: dict[str, dict],
) -> dict[str, dict]:
    data = extract_json_object_after_key(html, "tierLists")
    if not data:
        return {}
    out: dict[str, dict] = {}
    for tier_list in data.get("values") or []:
        for section in tier_list.get("tierSections") or []:
            if not isinstance(section, dict):
                continue
            tier_name = str(section.get("name") or "").strip()
            for item in section.get("staticDataItems") or []:
                if not isinstance(item, dict):
                    continue
                item_id = resolve_item_id(item, codex_by_slug=codex_by_slug)
                if not item_id:
                    continue
                notes = f"Mobalytics tier {tier_name}"
                merge_tier_entry(out, item_id, tier_name, notes)
    return out


def _build_title_to_archetype(title: str) -> str:
    cleaned = html_lib.unescape(title).strip()
    lower = cleaned.lower()
    for suffix in (" build", " deck"):
        if lower.endswith(suffix):
            lower = lower[: -len(suffix)]
            break
    if lower in BUILD_ARCHETYPE_MAP:
        return BUILD_ARCHETYPE_MAP[lower]
    if "body slam" in lower or "block" in lower:
        return "block body slam"
    if "exhaust" in lower:
        return "exhaust synergy"
    if "strength" in lower:
        return "strength scaling"
    return f"{lower} archetype"


def parse_character_archetypes(html: str) -> list[str]:
    archetypes: list[str] = []

    for match in re.finditer(r"<h2[^>]*>([^<]+)</h2>", html, re.I):
        title = html_lib.unescape(match.group(1)).strip()
        if re.search(r"\b(build|deck)\b", title, re.I) and "table of contents" not in title.lower():
            archetypes.append(_build_title_to_archetype(title))

    for match in re.finditer(
        r'"title":"([^"]+)"[^}]{0,120}"contentV2":(\{)',
        html,
    ):
        title = html_lib.unescape(match.group(1)).strip()
        if re.search(r"\b(build|deck)\b", title, re.I):
            archetypes.append(_build_title_to_archetype(title))

    # Keyword groups referenced in guide sections (e.g. strength scaling).
    for match in re.finditer(r'"groupId":"([^"]+)"', html):
        group = match.group(1).lower().replace("-", " ")
        if group in ("keywords", "overview", "contents"):
            continue
        if any(k in group for k in ("strength", "block", "exhaust", "blood", "strike")):
            archetypes.append(_build_title_to_archetype(group))

    # Dedupe preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for arch in archetypes:
        key = arch.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(arch)
    return ordered


def scrape(
    *,
    include_all_archetypes: bool = False,
    delay_sec: float = 0.75,
) -> dict[str, Any]:
    codex_by_slug, char_by_name = load_codex_indexes()

    cards_html = fetch_page(TIER_LIST_URLS["cards"])
    cards = parse_tier_lists(
        cards_html,
        codex_by_slug=codex_by_slug,
        char_by_name=char_by_name,
    )
    time.sleep(delay_sec)

    relics_html = fetch_page(TIER_LIST_URLS["relics"])
    relics = parse_relic_or_potion_page(relics_html, codex_by_slug=codex_by_slug)
    time.sleep(delay_sec)

    potions_html = fetch_page(TIER_LIST_URLS["potions"])
    potions = parse_relic_or_potion_page(potions_html, codex_by_slug=codex_by_slug)
    time.sleep(delay_sec)

    archetypes: list[str] = []
    archetypes_by_character: dict[str, list[str]] = {}
    guide_urls = CHARACTER_GUIDE_URLS if include_all_archetypes else {"ironclad": CHARACTER_GUIDE_URLS["ironclad"]}
    for character, url in guide_urls.items():
        guide_html = fetch_page(url)
        found = parse_character_archetypes(guide_html)
        archetypes_by_character[character] = found
        if character == "ironclad" or include_all_archetypes:
            archetypes.extend(found)
        time.sleep(delay_sec)

    # Dedupe archetypes list.
    arch_seen: set[str] = set()
    archetypes_out: list[str] = []
    for arch in archetypes:
        key = arch.lower()
        if key in arch_seen:
            continue
        arch_seen.add(key)
        archetypes_out.append(arch)

    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "mobalytics.gg",
        "cards": dict(sorted(cards.items())),
        "relics": dict(sorted(relics.items())),
        "potions": dict(sorted(potions.items())),
        "archetypes": archetypes_out,
        "archetypes_by_character": archetypes_by_character,
        "counts": {
            "cards": len(cards),
            "relics": len(relics),
            "potions": len(potions),
            "archetypes": len(archetypes_out),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output JSON path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--all-character-archetypes",
        action="store_true",
        help="Scrape archetypes from all character guides (default: ironclad only)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    payload = scrape(include_all_archetypes=args.all_character_archetypes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = payload.get("counts") or {}
    print(f"Wrote {args.output}")
    print(
        f"  cards={counts.get('cards', 0)} relics={counts.get('relics', 0)} "
        f"potions={counts.get('potions', 0)} archetypes={counts.get('archetypes', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
