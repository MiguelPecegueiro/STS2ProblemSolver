"""Learned enemy compendium - built from live combat, no spreadsheet required."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sts2_agent.knowledge import _normalize_name
from sts2_agent.scorer import _intent_damage_value, intent_threat_score

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
COMPENDIUM_PATH = DATA_DIR / "enemy_compendium.json"
OBSERVATIONS_PATH = DATA_DIR / "enemy_observations.jsonl"

_compendium_writes_enabled = True


def set_compendium_writes_enabled(enabled: bool) -> None:
    """Disable disk writes for parallel workers (avoids file conflicts)."""
    global _compendium_writes_enabled
    _compendium_writes_enabled = enabled


def compendium_writes_enabled() -> bool:
    return _compendium_writes_enabled


def configure_compendium_paths(
    data_dir: str | Path | None = None,
    *,
    writes_enabled: bool | None = None,
) -> None:
    """Optional per-instance compendium paths (defaults to shared data/)."""
    global DATA_DIR, COMPENDIUM_PATH, OBSERVATIONS_PATH
    if data_dir is not None:
        root = Path(data_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        DATA_DIR = root
        COMPENDIUM_PATH = DATA_DIR / "enemy_compendium.json"
        OBSERVATIONS_PATH = DATA_DIR / "enemy_observations.jsonl"
        _invalidate_kb_cache()
    if writes_enabled is not None:
        set_compendium_writes_enabled(writes_enabled)

BUFF_TAGS = frozenset(
    {"strength", "dexterity", "ritual", "artifact", "plated", "metallicize", "focus", "energized"}
)
DEBUFF_TAGS = frozenset(
    {"weak", "vulnerable", "frail", "poison", "daze", "hex", "doom", "stunned", "sleep", "wound"}
)
ROLE_TOKEN_RE = re.compile(
    r"(?:segment[_\s]*)?(front|middle|back|left|right|head|tail|core|body|wing|eye|mouth|leg)",
    re.I,
)
# API intent titles (STS2) -> semantic tags when description lacks keywords.
INTENT_TITLE_TAGS: dict[str, list[str]] = {
    "aggressive": ["attack"],
    "attack": ["attack"],
    "defend": ["block"],
    "defensive": ["block"],
    "block": ["block"],
    "heal": ["buff:heal"],
    "buff": ["buff:generic"],
    "debuff": ["debuff:generic"],
    "strategic": ["buff:generic"],
    "empower": ["buff:strength"],
    "enrage": ["buff:strength"],
    "strengthen": ["buff:strength"],
    "sleep": ["debuff:sleep"],
    "stun": ["debuff:stunned"],
    "skulk": ["buff:generic"],
    "curl": ["block"],
}

# Per combat: (run_key, entity_id) -> list of move_key observed this fight
_fight_history: dict[tuple[str, str], list[str]] = {}
_entity_names: dict[tuple[str, str], str] = {}
_entity_storage_keys: dict[tuple[str, str], str] = {}
# Last recorded move_key per entity (dedupe polls)
_last_turn_key: dict[tuple[str, str], str] = {}
# Last resolved for combat logging
_last_resolved: dict[tuple[str, str], dict[str, Any]] = {}


@dataclass
class ResolvedIntent:
    api_label: str
    move_key: str | None
    move_name: str | None
    damage: int
    block: int
    tags: list[str] = field(default_factory=list)
    source: str = "live"
    predicted_key: str | None = None
    prediction_match: bool | None = None


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def move_key(api_label: str, damage: int = 0, block: int = 0) -> str:
    label = (api_label or "unknown").strip().lower().replace(" ", "_")
    return f"{label}|d{damage}|b{block}"


def move_display_name(key: str) -> str:
    part = key.split("|", 1)[0]
    return part.replace("_", " ").title()


def entity_role_suffix(entity_id: str, display_name: str = "") -> str | None:
    """Role token from entity_id (segment front, head, etc.) when present."""
    eid = (entity_id or "").upper()
    if not eid:
        return None
    match = ROLE_TOKEN_RE.search(eid)
    if match:
        return match.group(1).lower()
    name_key = _normalize_name(display_name).replace(" ", "_").upper()
    if name_key and eid.startswith(name_key + "_"):
        tail = eid[len(name_key) + 1 :]
        parts = [p for p in tail.split("_") if p and not p.isdigit()]
        if len(parts) == 1 and parts[0].lower() not in ("0", "1", "2"):
            return parts[0].lower()
    return None


def compendium_base_key(enemy: dict) -> str:
    return _normalize_name(str(enemy.get("name") or "?")).replace(" ", "_")


def compendium_storage_key(enemy: dict, *, peers: list[dict] | None = None) -> str:
    """Stable compendium bucket: base name, or base/role, or base/slotN for twins."""
    base = compendium_base_key(enemy)
    entity_id = str(enemy.get("entity_id") or enemy.get("id") or "")
    role = entity_role_suffix(entity_id, str(enemy.get("name") or ""))
    if role:
        return f"{base}/{role}"

    peer_list = peers or []
    norm = _normalize_name(str(enemy.get("name") or ""))
    same_name = [
        e
        for e in peer_list
        if isinstance(e, dict)
        and int(e.get("hp") or 0) > 0
        and _normalize_name(str(e.get("name") or "")) == norm
    ]
    if len(same_name) > 1:
        ids = sorted(str(e.get("entity_id") or e.get("id") or "") for e in same_name)
        try:
            idx = ids.index(entity_id)
        except ValueError:
            idx = 0
        return f"{base}/slot{idx}"
    return base


def enemy_fight_label(enemy: dict, *, peers: list[dict] | None = None) -> str:
    """Human-readable label for logs (name + role/slot)."""
    name = str(enemy.get("name") or "?")
    key = compendium_storage_key(enemy, peers=peers)
    base = compendium_base_key(enemy)
    if key == base:
        return name
    suffix = key.split("/", 1)[-1].replace("_", " ").title()
    return f"{name} [{suffix}]"


def tags_include_debuff(tags: list[str]) -> bool:
    return any(str(t).startswith("debuff:") for t in tags)


def tags_include_buff(tags: list[str]) -> bool:
    return any(str(t).startswith("buff:") for t in tags)


def _living_peers(battle: dict | None, enemy: dict) -> list[dict]:
    if not battle:
        return [enemy]
    return [
        e
        for e in battle.get("enemies") or []
        if isinstance(e, dict) and int(e.get("hp") or 0) > 0
    ]


def _title_semantic_tags(api_label: str, intent: dict) -> list[str]:
    tags: list[str] = []
    title = api_label.strip().lower().replace(" ", "_")
    if title in INTENT_TITLE_TAGS:
        tags.extend(INTENT_TITLE_TAGS[title])
    itype = str(intent.get("type") or intent.get("intent_type") or "").strip().lower()
    if itype in INTENT_TITLE_TAGS:
        for t in INTENT_TITLE_TAGS[itype]:
            if t not in tags:
                tags.append(t)
    if itype == "debuff" and not tags_include_debuff(tags):
        tags.append("debuff:generic")
    if itype in ("buff", "power") and not tags_include_buff(tags):
        tags.append("buff:generic")
    return tags


def _extract_tags(intent: dict, api_label: str) -> list[str]:
    tags: list[str] = []
    tags.extend(_title_semantic_tags(api_label, intent))
    text = " ".join(
        str(intent.get(k) or "")
        for k in ("type", "title", "text", "description", "name")
    ).lower()
    text = f"{text} {api_label.lower()}"
    threat, label = intent_threat_score(intent)
    if "attack" in label or "attack" in text:
        if "attack" not in tags:
            tags.append("attack")
    if "block" in label or "defend" in text or "block" in text:
        if "block" not in tags:
            tags.append("block")
    for tag in BUFF_TAGS:
        if tag in text:
            buff = f"buff:{tag}"
            if buff not in tags:
                tags.append(buff)
    for tag in DEBUFF_TAGS:
        if tag in text:
            debuff = f"debuff:{tag}"
            if debuff not in tags:
                tags.append(debuff)
    if "debuff" in label and not tags_include_debuff(tags):
        tags.append("debuff:generic")
    if "buff" in label and not tags_include_buff(tags):
        tags.append("buff:generic")
    if not tags and threat > 0 and "attack" not in tags:
        if intent.get("damage") or _intent_damage_value(intent) > 0:
            tags.append("attack")
    return tags or ["unknown"]


def compact_enemy_intent(enemy: dict, *, peers: list[dict] | None = None) -> dict | None:
    """Compact intent snapshot for logging / decision state_snapshot."""
    if not isinstance(enemy, dict):
        return None
    intents = enemy.get("intents") or []
    if isinstance(enemy.get("intent"), dict):
        intents = [enemy["intent"]]
    if not intents or not isinstance(intents[0], dict):
        return None
    intent = intents[0]
    api_label = _intent_label(intent)
    damage = _intent_damage_value(intent)
    threat, tlabel = intent_threat_score(intent)
    if damage <= 0 and "attack" in tlabel:
        damage = int(threat) if threat > 5 else 0
    blk = int(enemy.get("block") or 0)
    tags = _extract_tags(intent, api_label)
    storage_key = compendium_storage_key(enemy, peers=peers)
    role = entity_role_suffix(
        str(enemy.get("entity_id") or enemy.get("id") or ""),
        str(enemy.get("name") or ""),
    )
    return {
        "entity_id": enemy.get("entity_id"),
        "name": enemy.get("name"),
        "compendium_key": storage_key,
        "role": role,
        "intent": api_label,
        "damage": damage,
        "block": blk,
        "tags": tags,
    }


def _intent_label(intent: Any) -> str:
    if isinstance(intent, dict):
        for key in ("title", "name", "move", "type", "text", "description"):
            val = intent.get(key)
            if val:
                return str(val).strip()
    return str(intent or "").strip()


def _run_key(state: dict) -> str:
    run = state.get("run") or {}
    return str(run.get("run_id") or run.get("seed") or "default")


def _run_id_from_state(state: dict | None) -> str | None:
    if not state:
        return None
    run = state.get("run") or {}
    rid = run.get("run_id")
    if rid:
        return str(rid)
    try:
        from sts2_agent.data_pipeline import get_pipeline

        p = get_pipeline()
        if p._run_active and p.run_id:
            return str(p.run_id)
    except Exception:
        pass
    return None


def _load_raw() -> dict[str, Any]:
    if not COMPENDIUM_PATH.exists():
        return {"version": 1, "enemies": {}}
    try:
        return json.loads(COMPENDIUM_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "enemies": {}}


def _save_raw(data: dict[str, Any]) -> None:
    if not _compendium_writes_enabled:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    COMPENDIUM_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _invalidate_kb_cache()


_kb_cache: Any = None


def _invalidate_kb_cache() -> None:
    global _kb_cache
    _kb_cache = None
    try:
        import sts2_agent.enemy_patterns as ep

        ep._kb = None  # noqa: SLF001
    except Exception:
        pass


class LearnedCompendiumKB:
    """Adapter so combat code can lookup enemies like the old pattern KB."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.by_name: dict[str, dict] = {}
        for key, entry in (raw.get("enemies") or {}).items():
            if isinstance(entry, dict):
                self.by_name[str(key).lower()] = entry

    def lookup(self, enemy_name: str) -> dict | None:
        norm = _normalize_name(enemy_name).replace(" ", "_")
        if norm in self.by_name:
            return self.by_name[norm]
        for key, entry in self.by_name.items():
            if key in norm or norm in key:
                return entry
        return None

    def lookup_enemy(
        self,
        enemy: dict,
        *,
        peers: list[dict] | None = None,
    ) -> dict | None:
        key = compendium_storage_key(enemy, peers=peers)
        if key in self.by_name:
            return self.by_name[key]
        base = compendium_base_key(enemy)
        if base in self.by_name:
            return self.by_name[base]
        norm = _normalize_name(str(enemy.get("name") or ""))
        if norm in self.by_name:
            return self.by_name[norm]
        for k, entry in self.by_name.items():
            if k.startswith(base + "/") or base.startswith(k):
                return entry
        return self.lookup(str(enemy.get("name") or ""))


def get_compendium_kb() -> LearnedCompendiumKB:
    global _kb_cache
    if _kb_cache is None:
        _kb_cache = LearnedCompendiumKB(_load_raw())
    return _kb_cache


def reload_compendium() -> LearnedCompendiumKB:
    _invalidate_kb_cache()
    return get_compendium_kb()


def _get_or_create_enemy(
    data: dict,
    storage_key: str,
    *,
    display_name: str,
    base_name: str | None = None,
    role: str | None = None,
    entity_id: str | None = None,
) -> dict:
    enemies = data.setdefault("enemies", {})
    if storage_key not in enemies:
        enemies[storage_key] = {
            "name": display_name,
            "base_name": base_name or display_name,
            "storage_key": storage_key,
            "role": role,
            "entity_id_sample": entity_id,
            "category": "unknown",
            "moves": {},
            "learned_cycle": [],
            "sequences": [],
            "fight_count": 0,
            "last_updated": _utc_now(),
        }
    entry = enemies[storage_key]
    entry["name"] = display_name
    if base_name:
        entry["base_name"] = base_name
    if role:
        entry["role"] = role
    if entity_id:
        entry["entity_id_sample"] = entity_id
    return entry


def _observe_intent(
    storage_key: str,
    intent: dict,
    *,
    display_name: str,
    base_name: str,
    role: str | None = None,
    entity_id: str | None = None,
    enemy_block: int = 0,
    run_id: str | None = None,
) -> str:
    api_label = _intent_label(intent)
    damage = _intent_damage_value(intent)
    threat, tlabel = intent_threat_score(intent)
    if damage <= 0 and "attack" in tlabel and threat > 0:
        damage = int(threat)
    tags = _extract_tags(intent, api_label)
    if "attack" in tags and damage <= 0:
        damage = max(damage, int(threat) if threat > 5 else 0)
    blk = int(enemy_block or 0)
    if "block" in tags and blk <= 0:
        blk = max(blk, int(threat) if "block" in tlabel else 0)

    key = move_key(api_label, damage, blk)
    data = _load_raw()
    entry = _get_or_create_enemy(
        data,
        storage_key,
        display_name=display_name,
        base_name=base_name,
        role=role,
        entity_id=entity_id,
    )
    moves = entry.setdefault("moves", {})
    if key not in moves:
        moves[key] = {
            "api_label": api_label,
            "damage": damage,
            "block": blk,
            "tags": tags,
            "seen_count": 0,
            "verified_runs": [],
        }
    moves[key]["seen_count"] = int(moves[key].get("seen_count") or 0) + 1
    moves[key]["api_label"] = api_label
    moves[key]["damage"] = damage
    moves[key]["block"] = blk
    moves[key]["tags"] = tags
    entry["last_updated"] = _utc_now()
    _save_raw(data)

    if not _compendium_writes_enabled:
        return key

    if run_id:
        OBSERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        obs = {
            "timestamp": _utc_now(),
            "run_id": run_id,
            "storage_key": storage_key,
            "enemy_name": display_name,
            "base_name": base_name,
            "role": role,
            "entity_id": entity_id,
            "move_key": key,
            "api_label": api_label,
            "damage": damage,
            "block": blk,
            "tags": tags,
        }
        with OBSERVATIONS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obs, ensure_ascii=False) + "\n")

    return key


def begin_combat_observation() -> None:
    _fight_history.clear()
    _entity_names.clear()
    _entity_storage_keys.clear()
    _last_turn_key.clear()
    _last_resolved.clear()


def record_enemy_intents_from_state(state: dict) -> None:
    """Record each new enemy intent this turn into fight history."""
    battle = state.get("battle") or {}
    run_key = _run_key(state)
    run_id = _run_id_from_state(state)
    peers = _living_peers(battle, {})

    for enemy in battle.get("enemies") or []:
        if int(enemy.get("hp") or 0) <= 0:
            continue
        name = str(enemy.get("name") or "?")
        entity_id = str(enemy.get("entity_id") or enemy.get("id") or name)
        key_entity = (run_key, entity_id)
        storage_key = compendium_storage_key(enemy, peers=peers)
        role = entity_role_suffix(entity_id, name)
        _entity_names[key_entity] = name
        _entity_storage_keys[key_entity] = storage_key

        intents = enemy.get("intents") or []
        if isinstance(enemy.get("intent"), dict):
            intents = [enemy["intent"]]
        if not intents or not isinstance(intents[0], dict):
            continue

        intent = intents[0]
        fight_label = enemy_fight_label(enemy, peers=peers)
        mk = _observe_intent(
            storage_key,
            intent,
            display_name=fight_label,
            base_name=name,
            role=role,
            entity_id=entity_id,
            enemy_block=int(enemy.get("block") or 0),
            run_id=run_id,
        )
        if _last_turn_key.get(key_entity) == mk:
            resolved, _ = resolve_enemy_intent(enemy, state)
            if resolved:
                _last_resolved[key_entity] = {
                    "api_label": resolved.api_label,
                    "move_name": resolved.move_name,
                    "move_key": resolved.move_key,
                    "damage": resolved.damage,
                    "tags": list(resolved.tags),
                    "source": resolved.source,
                    "predicted_move": resolved.predicted_key,
                    "prediction_match": resolved.prediction_match,
                }
            continue

        _last_turn_key[key_entity] = mk
        hist = _fight_history.setdefault(key_entity, [])
        hist.append(mk)

        resolved, _ = resolve_enemy_intent(enemy, state)
        if resolved and resolved.prediction_match and run_id and resolved.move_key:
            _mark_verified(storage_key, resolved.move_key, run_id)

        if resolved:
            _last_resolved[key_entity] = {
                "api_label": resolved.api_label,
                "move_name": resolved.move_name,
                "move_key": resolved.move_key,
                "damage": resolved.damage,
                "tags": list(resolved.tags),
                "source": resolved.source,
                "predicted_move": resolved.predicted_key,
                "prediction_match": resolved.prediction_match,
            }


def finalize_combat_observation(state: dict | None = None) -> None:
    """Merge fight histories into learned cycles."""
    run_id = _run_id_from_state(state) if state else None
    data = _load_raw()
    merged_any = False

    for key_entity, sequence in list(_fight_history.items()):
        if len(sequence) < 1:
            continue
        enemy_name = _entity_names.get(key_entity)
        storage_key = _entity_storage_keys.get(key_entity)
        if not enemy_name or not storage_key:
            continue

        role = None
        if "/" in storage_key:
            role = storage_key.split("/", 1)[1]
        label = (
            f"{enemy_name} [{role.replace('_', ' ').title()}]"
            if role
            else enemy_name
        )
        entry = _get_or_create_enemy(
            data,
            storage_key,
            display_name=label,
            base_name=enemy_name,
            role=role,
            entity_id=key_entity[1],
        )
        entry["fight_count"] = int(entry.get("fight_count") or 0) + 1

        seqs = entry.setdefault("sequences", [])
        seqs.append(
            {
                "run_id": run_id or "",
                "entity_id": key_entity[1],
                "moves": sequence,
            }
        )
        seqs[:] = seqs[-8:]

        entry["learned_cycle"] = _derive_cycle(seqs)
        entry["last_updated"] = _utc_now()
        merged_any = True

    if merged_any:
        _save_raw(data)

    _fight_history.clear()
    _entity_storage_keys.clear()
    _last_turn_key.clear()


def _derive_cycle(sequences: list[dict]) -> list[str]:
    """Pick best learned cycle from recent fight sequences."""
    if not sequences:
        return []
    moves_lists = [s.get("moves") or [] for s in sequences if s.get("moves")]
    if not moves_lists:
        return []
    longest = max(moves_lists, key=len)
    if len(longest) >= 2:
        return list(longest)
    return list(moves_lists[-1])


def _predict_next_key(entry: dict, history: list[str]) -> str | None:
    cycle = entry.get("learned_cycle") or []
    if not cycle:
        return None
    idx = len(history)
    if entry.get("pattern_kind") == "alternate" and len(history) >= 1:
        last = history[-1]
        if last in cycle:
            pos = cycle.index(last)
            return cycle[(pos + 1) % len(cycle)]
    return cycle[idx % len(cycle)]


def _move_stats(entry: dict, key: str | None) -> dict | None:
    if not key:
        return None
    return (entry.get("moves") or {}).get(key)


def _mark_verified(storage_key: str, mk: str, run_id: str) -> None:
    data = _load_raw()
    entry = (data.get("enemies") or {}).get(storage_key)
    if not entry:
        return
    move = (entry.get("moves") or {}).get(mk)
    if not move:
        return
    verified = move.setdefault("verified_runs", [])
    if run_id not in verified:
        verified.append(run_id)
        if len(verified) > 20:
            move["verified_runs"] = verified[-20:]
        _save_raw(data)


def resolve_enemy_intent(
    enemy: dict,
    state: dict | None = None,
) -> tuple[ResolvedIntent | None, list[str]]:
    reasons: list[str] = []
    intents = enemy.get("intents") or []
    if isinstance(enemy.get("intent"), dict):
        intents = [enemy["intent"]]
    if not intents or not isinstance(intents[0], dict):
        return None, reasons

    intent = intents[0]
    battle = (state or {}).get("battle") or {}
    peers = _living_peers(battle, enemy)
    label = enemy_fight_label(enemy, peers=peers)
    api_label = _intent_label(intent)
    damage = _intent_damage_value(intent)
    threat, tlabel = intent_threat_score(intent)
    if damage <= 0 and "attack" in tlabel:
        damage = int(threat) if threat > 5 else 0
    blk = int(enemy.get("block") or 0)
    tags = _extract_tags(intent, api_label)
    mk = move_key(api_label, damage, blk)

    kb = get_compendium_kb()
    entry = kb.lookup_enemy(enemy, peers=peers)

    history: list[str] = []
    predicted_key: str | None = None
    if state and entry:
        run_key = _run_key(state)
        entity_id = str(enemy.get("entity_id") or enemy.get("id") or label)
        history = list(_fight_history.get((run_key, entity_id), []))
        predicted_key = _predict_next_key(entry, history)

    known = _move_stats(entry, mk) if entry else None
    move_name = move_display_name(mk)
    tag_summary = ",".join(tags) if tags else "unknown"

    live_dmg = damage if "attack" in tags or damage > 0 else 0
    if live_dmg <= 0 and known and int(known.get("damage") or 0) > 0:
        if "attack" in (known.get("tags") or []):
            live_dmg = int(known["damage"])

    out_damage = live_dmg
    source = "live" if live_dmg > 0 else "learned"
    if not live_dmg and predicted_key:
        pred = _move_stats(entry, predicted_key) if entry else None
        if pred and int(pred.get("damage") or 0) > 0:
            out_damage = int(pred["damage"])
            source = "learned_predict"

    prediction_match = None
    if predicted_key and mk:
        prediction_match = predicted_key == mk
        pred_move = _move_stats(entry, predicted_key) if entry else None
        pred_tags = ",".join((pred_move or {}).get("tags") or [])
        if prediction_match:
            reasons.append(
                f"{label}: learned '{api_label}' [{tag_summary}] -> {move_name} (matched)"
            )
        else:
            reasons.append(
                f"{label}: '{api_label}' [{tag_summary}] dmg={out_damage} "
                f"(predicted {move_display_name(predicted_key)} [{pred_tags}])"
            )
    elif known:
        reasons.append(
            f"{label}: compendium '{api_label}' dmg={out_damage} blk={blk} "
            f"tags={tag_summary} (seen {known.get('seen_count', 0)}x)"
        )
    elif live_dmg > 0:
        reasons.append(f"{label}: live '{api_label}' [{tag_summary}] dmg={live_dmg} (learning)")
    elif predicted_key:
        pred_move = _move_stats(entry, predicted_key) if entry else None
        pred_tags = ",".join((pred_move or {}).get("tags") or [])
        reasons.append(
            f"{label}: predict {move_display_name(predicted_key)} [{pred_tags}]"
        )
    elif tags_include_debuff(tags):
        reasons.append(f"{label}: debuff intent '{api_label}' [{tag_summary}]")
    elif tags_include_buff(tags):
        reasons.append(f"{label}: buff intent '{api_label}' [{tag_summary}]")

    return (
        ResolvedIntent(
            api_label=api_label,
            move_key=mk,
            move_name=move_name,
            damage=out_damage,
            block=blk,
            tags=tags,
            source=source,
            predicted_key=predicted_key,
            prediction_match=prediction_match,
        ),
        reasons,
    )


def move_incoming_damage(move: dict | None) -> int:
    if not move:
        return 0
    if "attack" in (move.get("tags") or []) or int(move.get("damage") or 0) > 0:
        return int(move.get("damage") or 0)
    return 0


def player_damage_taken_multiplier(state: dict | None) -> tuple[float, list[str]]:
    """Scale expected incoming damage when the player has Weak / Vulnerable."""
    if not state:
        return 1.0, []
    player = state.get("player") or {}
    mult = 1.0
    reasons: list[str] = []
    for status in player.get("status") or player.get("powers") or []:
        if isinstance(status, dict):
            name = str(status.get("id") or status.get("name") or status.get("power") or "")
        else:
            name = str(status)
        low = name.lower()
        if "weak" in low and mult == 1.0:
            mult *= 1.25
            reasons.append("player Weak - expect +25% damage taken")
        if "vulnerable" in low:
            mult *= 1.5
            reasons.append("player Vulnerable - expect +50% damage taken")
        if "frail" in low:
            reasons.append("player Frail - block gains reduced")
    return mult, reasons


def assess_combat_debuff_pressure(
    enemies: list[dict],
    state: dict | None,
) -> tuple[bool, list[str]]:
    """True when compendium predicts player debuffs soon or player is already debuffed."""
    reasons: list[str] = []
    if not state:
        return False, reasons

    battle = state.get("battle") or {}
    peers = _living_peers(battle, {})
    kb = get_compendium_kb()
    run_key = _run_key(state)
    predicts_debuff = False

    for enemy in enemies:
        if int(enemy.get("hp") or 0) <= 0:
            continue
        label = enemy_fight_label(enemy, peers=peers)
        entry = kb.lookup_enemy(enemy, peers=peers)
        if not entry:
            continue
        entity_id = str(enemy.get("entity_id") or enemy.get("id") or label)
        history = list(_fight_history.get((run_key, entity_id), []))
        resolved, _ = resolve_enemy_intent(enemy, state)
        if resolved and resolved.move_key:
            history = history + [resolved.move_key]
        pred_key = _predict_next_key(entry, history)
        pred_move = _move_stats(entry, pred_key)
        if pred_move and tags_include_debuff(list(pred_move.get("tags") or [])):
            predicts_debuff = True
            reasons.append(
                f"{label}: next turn may debuff ({move_display_name(pred_key or '')}, "
                f"{','.join(pred_move.get('tags') or [])})"
            )
        if resolved and tags_include_debuff(resolved.tags):
            reasons.append(f"{label}: debuff intent now ({','.join(resolved.tags)})")

    _, player_reasons = player_damage_taken_multiplier(state)
    reasons.extend(player_reasons)
    player_debuffed = bool(player_reasons)
    return predicts_debuff or player_debuffed, reasons


def enrich_incoming_damage(
    enemies: list[dict],
    state: dict | None = None,
) -> tuple[int, int, list[str]]:
    reasons: list[str] = []
    this_turn = 0
    next_turn = 0
    battle = (state or {}).get("battle") or {}
    peers = _living_peers(battle, {})
    kb = get_compendium_kb()
    run_key = _run_key(state) if state else "default"

    for enemy in enemies:
        if int(enemy.get("hp") or 0) <= 0:
            continue
        label = enemy_fight_label(enemy, peers=peers)
        resolved, r = resolve_enemy_intent(enemy, state)
        reasons.extend(r)
        if not resolved:
            continue

        if resolved.damage > 0:
            this_turn += resolved.damage

        entry = kb.lookup_enemy(enemy, peers=peers)
        if not entry:
            continue

        entity_id = str(enemy.get("entity_id") or enemy.get("id") or label)
        history = list(_fight_history.get((run_key, entity_id), []))
        if resolved.move_key:
            history = history + [resolved.move_key]

        pred_key = _predict_next_key(entry, history)
        pred_move = _move_stats(entry, pred_key)
        if pred_move:
            nxt = move_incoming_damage(pred_move)
            if nxt > 0:
                next_turn += nxt
                reasons.append(
                    f"{label}: learned next-turn ~{nxt} ({move_display_name(pred_key or '')})"
                )
            elif tags_include_debuff(list(pred_move.get("tags") or [])):
                reasons.append(
                    f"{label}: learned next-turn debuff ({move_display_name(pred_key or '')}, "
                    f"{','.join(pred_move.get('tags') or [])})"
                )

    dmg_mult, mult_reasons = player_damage_taken_multiplier(state)
    reasons.extend(mult_reasons)
    if dmg_mult > 1.0:
        this_turn = int(this_turn * dmg_mult)
        next_turn = int(next_turn * dmg_mult)

    return this_turn, next_turn, reasons


def last_resolved_intents(state: dict) -> dict[str, dict[str, Any]]:
    run_key = _run_key(state)
    out: dict[str, dict[str, Any]] = {}
    for (rk, entity_id), data in _last_resolved.items():
        if rk == run_key:
            out[entity_id] = data
    return out


def clear_combat_history(state: dict | None = None) -> None:
    if state is None:
        _fight_history.clear()
        _last_turn_key.clear()
        _last_resolved.clear()
        return
    run_key = _run_key(state)
    for store in (_fight_history, _last_turn_key, _last_resolved):
        for k in [k for k in store if k[0] == run_key]:
            del store[k]


# Back-compat names used by combat / dashboard
def get_enemy_pattern_kb() -> LearnedCompendiumKB:
    return get_compendium_kb()


def predict_enemy_move(enemy: dict, state: dict | None = None) -> tuple[dict | None, list[str]]:
    battle = (state or {}).get("battle") or {}
    peers = _living_peers(battle, enemy)
    label = enemy_fight_label(enemy, peers=peers)
    entry = get_compendium_kb().lookup_enemy(enemy, peers=peers)
    if not entry:
        return None, [f"{label}: not in compendium yet"]
    run_key = _run_key(state) if state else "default"
    entity_id = str(enemy.get("entity_id") or enemy.get("id") or label)
    history = list(_fight_history.get((run_key, entity_id), []))
    pred_key = _predict_next_key(entry, history)
    if not pred_key:
        return None, [f"{label}: no learned cycle yet"]
    move = _move_stats(entry, pred_key) or {}
    move["name"] = move_display_name(pred_key)
    tags = ",".join(move.get("tags") or [])
    return move, [
        f"{label}: predict {move['name']} [{tags}] "
        f"(cycle len {len(entry.get('learned_cycle') or [])})"
    ]


def group_compendium_by_encounter(enemies: dict[str, dict]) -> dict[str, list[str]]:
    """Group storage keys by base_name for dashboard encounter view."""
    groups: dict[str, list[str]] = {}
    for key, entry in enemies.items():
        if not isinstance(entry, dict):
            continue
        base = str(entry.get("base_name") or key.split("/")[0])
        base_norm = _normalize_name(base)
        groups.setdefault(base_norm, []).append(key)
    for base in groups:
        groups[base].sort()
    return groups
