"""Load and cache Spire Codex + community stats for agent decisions."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

CODEX_BASE_URL = "https://spire-codex.com"
REPLAYS_BASE_URL = "https://sts2replays.com"
CACHE_MAX_AGE_SEC = 24 * 60 * 60
REQUEST_HEADERS = {"User-Agent": "STS2Agent/0.1 (local rule-based bot)"}
CODEX_ENDPOINTS = ("cards", "relics", "monsters", "encounters", "potions")
# Endpoints tried for sts2replays community stats (no public API documented as of 2026).
REPLAYS_STATS_URLS = (
    f"{REPLAYS_BASE_URL}/api/stats/cards",
    f"{REPLAYS_BASE_URL}/api/v1/stats/cards",
    f"{REPLAYS_BASE_URL}/api/cards/stats",
)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
CACHE_DIR = PROJECT_ROOT / "cache"
KNOWLEDGE_CACHE_PATH = CACHE_DIR / "knowledge.json"
COMMUNITY_CACHE_PATH = CACHE_DIR / "community_stats.json"
EXPERT_KNOWLEDGE_PATH = CACHE_DIR / "expert_knowledge.json"

EXPERT_TIER_BONUS = {"S": 40.0, "A": 25.0, "B": 10.0, "C": 0.0, "D": -25.0}

PLAYABLE_CHARACTERS = (
    "ironclad",
    "silent",
    "defect",
    "necrobinder",
    "regent",
)


class KnowledgeBase:
    """In-memory indexes built from cached Codex + community data."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.cards_by_id: dict[str, dict] = {}
        self.cards_by_name: dict[str, dict] = {}
        self.relics_by_id: dict[str, dict] = {}
        self.relics_by_name: dict[str, dict] = {}
        self.monsters_by_id: dict[str, dict] = {}
        self.monsters_by_name: dict[str, dict] = {}
        self.potions_by_id: dict[str, dict] = {}
        self.potions_by_name: dict[str, dict] = {}
        self.encounters: list[dict] = raw.get("encounters") or []
        self.community_cards: dict[str, dict] = {}
        expert = raw.get("expert") or {}
        self.expert_cards: dict[str, dict] = dict(expert.get("cards") or {})
        self.expert_relics: dict[str, dict] = dict(expert.get("relics") or {})
        self.expert_potions: dict[str, dict] = dict(expert.get("potions") or {})
        self.expert_archetypes: list[str] = list(expert.get("archetypes") or [])
        self.expert_archetypes_by_character: dict[str, list[str]] = dict(
            expert.get("archetypes_by_character") or {}
        )
        self._build_indexes()

    def _build_indexes(self) -> None:
        for card in self.raw.get("cards") or []:
            if not isinstance(card, dict):
                continue
            cid = str(card.get("id") or "").upper()
            name = str(card.get("name") or "")
            if cid:
                self.cards_by_id[cid] = card
            if name:
                self.cards_by_name[_normalize_name(name)] = card

        for relic in self.raw.get("relics") or []:
            if not isinstance(relic, dict):
                continue
            rid = str(relic.get("id") or "").upper()
            name = str(relic.get("name") or "")
            if rid:
                self.relics_by_id[rid] = relic
            if name:
                self.relics_by_name[_normalize_name(name)] = relic

        for monster in self.raw.get("monsters") or []:
            if not isinstance(monster, dict):
                continue
            mid = str(monster.get("id") or "").upper()
            name = str(monster.get("name") or "")
            if mid:
                self.monsters_by_id[mid] = monster
            if name:
                self.monsters_by_name[_normalize_name(name)] = monster

        for potion in self.raw.get("potions") or []:
            if not isinstance(potion, dict):
                continue
            pid = str(potion.get("id") or "").upper()
            name = str(potion.get("name") or "")
            if pid:
                self.potions_by_id[pid] = potion
            if name:
                self.potions_by_name[_normalize_name(name)] = potion

        community = self.raw.get("community") or {}
        for entry in community.get("cards") or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("card") or entry.get("id")
            if name:
                self.community_cards[_normalize_name(str(name))] = entry

    def lookup_card(self, name_or_id: str | None) -> dict | None:
        if not name_or_id:
            return None
        key = _normalize_name(name_or_id)
        if key in self.cards_by_name:
            return self.cards_by_name[key]
        upper = str(name_or_id).upper().replace(" ", "_")
        return self.cards_by_id.get(upper)

    def lookup_relic(self, name_or_id: str | None) -> dict | None:
        if not name_or_id:
            return None
        key = _normalize_name(name_or_id)
        if key in self.relics_by_name:
            return self.relics_by_name[key]
        upper = str(name_or_id).upper().replace(" ", "_")
        return self.relics_by_id.get(upper)

    def lookup_potion(self, name_or_id: str | None) -> dict | None:
        if not name_or_id:
            return None
        key = _normalize_name(name_or_id)
        if key in self.potions_by_name:
            return self.potions_by_name[key]
        upper = str(name_or_id).upper().replace(" ", "_")
        return self.potions_by_id.get(upper)

    def community_win_rate(self, card_name: str) -> float | None:
        entry = self.community_cards.get(_normalize_name(card_name))
        if not entry:
            return None
        for field in ("win_rate", "winRate", "win_pct", "winPct"):
            val = entry.get(field)
            if val is not None:
                rate = float(val)
                return rate / 100.0 if rate > 1.0 else rate
        return None

    def community_pick_rate(self, card_name: str) -> float | None:
        entry = self.community_cards.get(_normalize_name(card_name))
        if not entry:
            return None
        for field in ("pick_rate", "pickRate", "pick_pct", "pickPct"):
            val = entry.get(field)
            if val is not None:
                rate = float(val)
                return rate / 100.0 if rate > 1.0 else rate
        return None

    def _expert_entry(
        self,
        name_or_id: str | None,
        bucket: dict[str, dict],
    ) -> dict | None:
        if not name_or_id or not bucket:
            return None
        upper = str(name_or_id).upper().replace(" ", "_")
        if upper in bucket:
            return bucket[upper]
        codex = self.lookup_card(name_or_id)
        if codex:
            cid = str(codex.get("id") or "").upper()
            if cid in bucket:
                return bucket[cid]
        return None

    def expert_card_tier(self, name_or_id: str | None) -> str | None:
        entry = self._expert_entry(name_or_id, self.expert_cards)
        if not entry:
            return None
        tier = str(entry.get("tier") or "").upper()
        return tier or None

    def expert_card_bonus(self, name_or_id: str | None) -> tuple[float, str | None]:
        tier = self.expert_card_tier(name_or_id)
        if not tier:
            return 0.0, None
        bonus = float(EXPERT_TIER_BONUS.get(tier.upper(), 0.0))
        return bonus, tier

    def expert_card_notes(self, name_or_id: str | None) -> str | None:
        entry = self._expert_entry(name_or_id, self.expert_cards)
        if not entry:
            return None
        notes = str(entry.get("notes") or "").strip()
        return notes or None

    def character_archetypes(self, character: str | None) -> list[str]:
        if not character:
            return list(self.expert_archetypes)
        key = str(character).lower().replace("the ", "").strip()
        if key in self.expert_archetypes_by_character:
            return list(self.expert_archetypes_by_character[key])
        return list(self.expert_archetypes)


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace("_", " ").replace("-", " ")


def _cache_is_fresh(path: Path, max_age_sec: int = CACHE_MAX_AGE_SEC) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < max_age_sec


def _fetch_json(url: str, timeout: float = 60.0) -> Any:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _fetch_codex_bundle() -> dict[str, Any]:
    """Fetch all Codex endpoints (respect ~60 req/min - sequential with small delay)."""
    bundle: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "spire-codex.com",
    }
    for idx, endpoint in enumerate(CODEX_ENDPOINTS):
        url = f"{CODEX_BASE_URL}/api/{endpoint}"
        logger.info("Fetching Codex %s ...", endpoint)
        data = _fetch_json(url)
        bundle[endpoint] = data
        if idx < len(CODEX_ENDPOINTS) - 1:
            time.sleep(1.1)
    bundle["cards"] = bundle.get("cards") or []
    return bundle


def _parse_community_payload(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("cards", "data", "results", "items"):
            val = data.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
    return []


def _fetch_community_stats() -> dict[str, Any]:
    """Try sts2replays endpoints; return empty card list if unavailable."""
    result: dict[str, Any] = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "sts2replays.com",
        "cards": [],
        "available": False,
    }
    for url in REPLAYS_STATS_URLS:
        try:
            logger.info("Trying community stats: %s", url)
            data = _fetch_json(url, timeout=30.0)
            cards = _parse_community_payload(data)
            if cards:
                result["cards"] = cards
                result["available"] = True
                result["url"] = url
                logger.info("Loaded %d community card stats", len(cards))
                return result
        except requests.RequestException as exc:
            logger.debug("Community stats unavailable at %s: %s", url, exc)

    logger.warning(
        "Community stats from sts2replays.com unavailable (no public API found). "
        "Card rewards will use Codex data only. Place stats at %s to override.",
        COMMUNITY_CACHE_PATH,
    )
    return result


def refresh_cache(force: bool = False) -> Path:
    """Download Codex + community data and write cache files."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if force or not _cache_is_fresh(KNOWLEDGE_CACHE_PATH):
        codex = _fetch_codex_bundle()
        KNOWLEDGE_CACHE_PATH.write_text(
            json.dumps(codex, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Wrote Codex cache (%d cards, %d potions)",
            len(codex.get("cards") or []),
            len(codex.get("potions") or []),
        )
    else:
        logger.info("Codex cache is fresh: %s", KNOWLEDGE_CACHE_PATH)

    if force or not _cache_is_fresh(COMMUNITY_CACHE_PATH):
        community = _fetch_community_stats()
        COMMUNITY_CACHE_PATH.write_text(
            json.dumps(community, ensure_ascii=False), encoding="utf-8"
        )
    else:
        logger.info("Community cache is fresh: %s", COMMUNITY_CACHE_PATH)

    return KNOWLEDGE_CACHE_PATH


def load_knowledge(force_refresh: bool = False) -> KnowledgeBase:
    """Load knowledge from cache, refreshing from APIs when stale or missing."""
    if force_refresh or not _cache_is_fresh(KNOWLEDGE_CACHE_PATH):
        try:
            refresh_cache(force=force_refresh)
        except requests.RequestException as exc:
            logger.error("Failed to refresh knowledge cache: %s", exc)
            if not KNOWLEDGE_CACHE_PATH.exists():
                raise

    codex_raw = json.loads(KNOWLEDGE_CACHE_PATH.read_text(encoding="utf-8"))
    merged: dict[str, Any] = {
        "cards": codex_raw.get("cards") or [],
        "relics": codex_raw.get("relics") or [],
        "monsters": codex_raw.get("monsters") or [],
        "encounters": codex_raw.get("encounters") or [],
        "potions": codex_raw.get("potions") or [],
        "community": {"cards": []},
    }

    if COMMUNITY_CACHE_PATH.exists():
        community_raw = json.loads(COMMUNITY_CACHE_PATH.read_text(encoding="utf-8"))
        merged["community"] = community_raw
    elif not force_refresh:
        pass

    if EXPERT_KNOWLEDGE_PATH.exists():
        try:
            merged["expert"] = json.loads(EXPERT_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load expert knowledge: %s", exc)
            merged["expert"] = {}
    else:
        merged["expert"] = {}

    return KnowledgeBase(merged)


# Module-level singleton for handlers.
_kb: KnowledgeBase | None = None


def get_knowledge(force_refresh: bool = False) -> KnowledgeBase:
    global _kb
    if _kb is None or force_refresh:
        _kb = load_knowledge(force_refresh=force_refresh)
    return _kb
