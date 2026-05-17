"""Hand card semantic encoding (Spire Codex)."""

from __future__ import annotations

import numpy as np

from training.features import (
    CARD_FEAT_DIM,
    FEATURE_DIM,
    encode_hand_card_features,
    encode_snapshot,
)


class _StubKB:
    def lookup_card(self, name_or_id: str | None):
        key = str(name_or_id or "").lower()
        if "bash" in key:
            return {
                "id": "BASH",
                "type_key": "attack",
                "cost": 2,
                "damage": 8,
            }
        if "defend" in key:
            return {"id": "DEFEND", "type_key": "skill", "cost": 1, "block": 5}
        if "offering" in key:
            return {"id": "OFFERING", "type_key": "skill", "cost": 0}
        return None


def test_unknown_card_is_zeros() -> None:
    vec = encode_hand_card_features({"name": "NotARealCard"}, _StubKB())
    assert vec.shape == (CARD_FEAT_DIM,)
    assert np.all(vec == 0)


def test_bash_encoding() -> None:
    vec = encode_hand_card_features(
        {"name": "Bash", "cost": 2, "can_play": True, "is_upgraded": False},
        _StubKB(),
    )
    assert vec[0] == 0.5  # cost 2/4
    assert vec[1] == 1.0  # attack
    assert vec[2] == 0.0  # skill
    assert vec[4] == 0.0  # not upgraded
    assert vec[5] == 1.0  # can_play


def test_snapshot_hand_slot_uses_semantic_features() -> None:
    snap = {
        "player_hp": 50,
        "player_max_hp": 80,
        "player_block": 0,
        "player_energy": 3,
        "hand": [{"name": "Bash", "cost": 2, "type": "attack", "can_play": True}],
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "relics": [],
        "potions": [],
        "status_effects": [],
        "enemies": [],
    }
    vec = encode_snapshot(snap, state_type="monster")
    assert vec.shape == (FEATURE_DIM,)
    hand_base = 8 + 5 + 15 + 3 + 16  # player + pile + state + run + status
    assert vec[hand_base + 1] == 1.0  # is_attack on first hand slot
