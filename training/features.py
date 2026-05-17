"""Convert decision state_snapshot dicts into fixed-length float32 feature vectors."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Layout constants (also stored in model_config.json)
MAX_HAND = 10
MAX_ENEMIES = 5
MAX_RELICS = 20
MAX_POTIONS = 3
MAX_STATUS = 8
CARD_FEAT_DIM = 6
ENEMY_FEAT_DIM = 8
RELIC_BUCKETS = 32
POTION_BUCKETS = 16
STATUS_BUCKETS = 16
ENEMY_ID_BUCKETS = 32
MAX_CARD_COST_NORM = 4.0

STATE_TYPES = (
    "monster",
    "elite",
    "boss",
    "hand_select",
    "map",
    "card_reward",
    "rewards",
    "treasure",
    "rest_site",
    "shop",
    "event",
    "card_select",
    "menu",
    "game_over",
    "unknown",
)
STATE_TYPE_TO_IDX = {name: i for i, name in enumerate(STATE_TYPES)}

PLAYER_FEAT_DIM = 8
PILE_FEAT_DIM = 5  # draw attack/block ratios, high-value flag, next-turn block/damage est.

# player + pile + state_onehot + run(3) + status + hand + enemies + relics + potions
FEATURE_DIM = (
    PLAYER_FEAT_DIM
    + PILE_FEAT_DIM
    + len(STATE_TYPES)
    + 3
    + STATUS_BUCKETS
    + MAX_HAND * CARD_FEAT_DIM
    + MAX_ENEMIES * ENEMY_FEAT_DIM
    + RELIC_BUCKETS
    + POTION_BUCKETS
)


def feature_layout() -> dict[str, Any]:
    """Describe vector layout for model_config.json."""
    return {
        "feature_dim": FEATURE_DIM,
        "max_hand": MAX_HAND,
        "max_enemies": MAX_ENEMIES,
        "max_relics": MAX_RELICS,
        "max_potions": MAX_POTIONS,
        "card_feat_dim": CARD_FEAT_DIM,
        "enemy_feat_dim": ENEMY_FEAT_DIM,
        "player_feat_dim": PLAYER_FEAT_DIM,
        "pile_feat_dim": PILE_FEAT_DIM,
        "state_types": list(STATE_TYPES),
        "hand_card_features": [
            "cost_norm",
            "is_attack",
            "is_skill",
            "is_power",
            "is_upgraded",
            "can_play",
        ],
        "pile_features": [
            "attack_ratio_in_draw",
            "block_ratio_in_draw",
            "high_value_cards_in_draw",
            "expected_block_next_turn",
            "expected_damage_next_turn",
        ],
    }


def _hash_bucket(text: str, buckets: int) -> int:
    if not text:
        return 0
    digest = hashlib.md5(text.strip().upper().encode("utf-8")).hexdigest()
    return int(digest, 16) % buckets


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_card_cost(card: dict, codex: dict) -> float:
    """Energy cost in [0, 1] with max 4 (X-cost → 1.0)."""
    from sts2_agent.scorer import _card_cost

    raw = _card_cost(card)
    if raw >= 99:
        codex_cost = codex.get("cost") or codex.get("energy_cost") or codex.get("energy")
        if codex_cost is not None:
            try:
                raw = int(codex_cost)
            except (TypeError, ValueError):
                return 1.0
        else:
            return 1.0
    return min(max(float(raw), 0.0) / MAX_CARD_COST_NORM, 1.0)


def _codex_type_flags(codex: dict) -> tuple[float, float, float]:
    from sts2_agent.scorer import _is_attack_type

    type_key = str(codex.get("type_key") or codex.get("type") or "").lower()
    is_attack = 1.0 if _is_attack_type(codex) else 0.0
    is_skill = 1.0 if "skill" in type_key else 0.0
    is_power = 1.0 if "power" in type_key else 0.0
    return is_attack, is_skill, is_power


def encode_hand_card_features(card: dict, kb: Any) -> np.ndarray:
    """Six semantic features per hand card (Spire Codex + live can_play / upgraded)."""
    name_or_id = str(card.get("id") or card.get("name") or "")
    codex = kb.lookup_card(name_or_id) if name_or_id else None
    if not codex:
        return np.zeros(CARD_FEAT_DIM, dtype=np.float32)

    is_attack, is_skill, is_power = _codex_type_flags(codex)
    is_upgraded = 1.0 if card.get("is_upgraded") in (True, 1, "true", "True") else 0.0
    can_play = 1.0
    if card.get("can_play") is False or card.get("playable") is False:
        can_play = 0.0

    return np.array(
        [
            _normalize_card_cost(card, codex),
            is_attack,
            is_skill,
            is_power,
            is_upgraded,
            can_play,
        ],
        dtype=np.float32,
    )


def encode_snapshot(
    snapshot: dict | None,
    *,
    state_type: str = "",
    floor: int = 0,
    act: int = 1,
    immediate_reward: float = 0.0,
) -> np.ndarray:
    """Encode one decision row into a float32 vector of shape (FEATURE_DIM,)."""
    snap = snapshot or {}
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    offset = 0

    hp = _safe_float(snap.get("player_hp"))
    max_hp = max(_safe_float(snap.get("player_max_hp"), 1.0), 1.0)
    block = _safe_float(snap.get("player_block"))
    energy = _safe_float(snap.get("player_energy"))
    hand = [c for c in (snap.get("hand") or []) if isinstance(c, dict)]
    relics = [r for r in (snap.get("relics") or []) if r]
    potions = [p for p in (snap.get("potions") or []) if p]
    statuses = [s for s in (snap.get("status_effects") or []) if s]
    enemies = [e for e in (snap.get("enemies") or []) if isinstance(e, dict)]

    from sts2_agent.knowledge import get_knowledge

    kb = get_knowledge()

    player = np.array(
        [
            hp / max_hp,
            min(block / max_hp, 3.0),
            min(energy / 10.0, 1.0),
            min(len(hand) / MAX_HAND, 1.0),
            min(_safe_float(snap.get("draw_pile_count")) / 50.0, 1.0),
            min(_safe_float(snap.get("discard_pile_count")) / 50.0, 1.0),
            min(len(potions) / MAX_POTIONS, 1.0),
            min(len(relics) / MAX_RELICS, 1.0),
        ],
        dtype=np.float32,
    )
    vec[offset : offset + PLAYER_FEAT_DIM] = player
    offset += PLAYER_FEAT_DIM

    pile = np.array(
        [
            float(np.clip(_safe_float(snap.get("attack_ratio_in_draw")), 0.0, 1.0)),
            float(np.clip(_safe_float(snap.get("block_ratio_in_draw")), 0.0, 1.0)),
            float(np.clip(_safe_float(snap.get("high_value_cards_in_draw")), 0.0, 1.0)),
            min(_safe_float(snap.get("expected_block_next_turn")) / 30.0, 1.0),
            min(_safe_float(snap.get("expected_damage_next_turn")) / 30.0, 1.0),
        ],
        dtype=np.float32,
    )
    vec[offset : offset + PILE_FEAT_DIM] = pile
    offset += PILE_FEAT_DIM

    st_idx = STATE_TYPE_TO_IDX.get(str(state_type or "").lower(), STATE_TYPE_TO_IDX["unknown"])
    vec[offset + st_idx] = 1.0
    offset += len(STATE_TYPES)

    run_ctx = np.array(
        [
            min(floor / 50.0, 1.0),
            min(act / 3.0, 1.0),
            float(np.clip(immediate_reward / 30.0, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )
    vec[offset : offset + 3] = run_ctx
    offset += 3

    for status in statuses[:MAX_STATUS]:
        bucket = _hash_bucket(str(status), STATUS_BUCKETS)
        vec[offset + bucket] = 1.0
    offset += STATUS_BUCKETS

    for slot in range(MAX_HAND):
        base = offset + slot * CARD_FEAT_DIM
        if slot >= len(hand):
            continue
        vec[base : base + CARD_FEAT_DIM] = encode_hand_card_features(hand[slot], kb)
    offset += MAX_HAND * CARD_FEAT_DIM

    for slot in range(MAX_ENEMIES):
        base = offset + slot * ENEMY_FEAT_DIM
        if slot >= len(enemies):
            continue
        enemy = enemies[slot]
        e_hp = _safe_float(enemy.get("hp"))
        e_max = max(_safe_float(enemy.get("max_hp"), e_hp), 1.0)
        e_block = _safe_float(enemy.get("block"))
        intent = str(enemy.get("intent") or "").lower()
        intent_val = _safe_float(enemy.get("intent_value"))
        tags = [str(t).lower() for t in (enemy.get("intent_tags") or [])]
        eid = str(enemy.get("entity_id") or enemy.get("id") or enemy.get("name") or "")

        vec[base] = 1.0
        vec[base + 1] = e_hp / e_max if e_max else 0.0
        vec[base + 2] = min(e_block / e_max, 3.0) if e_max else 0.0
        vec[base + 3] = min(intent_val / 30.0, 1.0)
        vec[base + 4] = 1.0 if "attack" in intent or "attack" in tags else 0.0
        vec[base + 5] = 1.0 if "block" in intent or any("block" in t for t in tags) else 0.0
        vec[base + 6] = 1.0 if any(t.startswith("debuff") for t in tags) else 0.0
        vec[base + 7] = _hash_bucket(eid, ENEMY_ID_BUCKETS) / ENEMY_ID_BUCKETS
    offset += MAX_ENEMIES * ENEMY_FEAT_DIM

    for relic in relics[:MAX_RELICS]:
        bucket = _hash_bucket(str(relic), RELIC_BUCKETS)
        vec[offset + bucket] = 1.0
    offset += RELIC_BUCKETS

    for potion in potions[:MAX_POTIONS]:
        bucket = _hash_bucket(str(potion), POTION_BUCKETS)
        vec[offset + bucket] = 1.0

    return vec


def encode_snapshot_batch(rows: list[dict]) -> np.ndarray:
    """Stack feature vectors for a list of decision dicts."""
    out = np.zeros((len(rows), FEATURE_DIM), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i] = encode_snapshot(
            row.get("state_snapshot"),
            state_type=str(row.get("state_type") or ""),
            floor=int(row.get("floor") or 0),
            act=int(row.get("act") or 1),
            immediate_reward=float(row.get("immediate_reward") or 0.0),
        )
    return out
