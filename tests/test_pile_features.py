"""Draw-pile composition features for policy snapshots."""

from __future__ import annotations

import numpy as np

from sts2_agent.data_pipeline import build_state_snapshot
from sts2_agent.pile_odds import draw_pile_feature_summary
from training.features import FEATURE_DIM, PILE_FEAT_DIM, encode_snapshot


class _StubKB:
    def lookup_card(self, name: str):
        n = str(name).lower()
        if "bash" in n or "strike" in n:
            return {"type_key": "attack", "damage": 8}
        if "defend" in n or "block" in n:
            return {"type_key": "skill", "block": 5}
        return {}


def test_draw_pile_feature_summary_ratios() -> None:
    player = {
        "energy": 3,
        "max_energy": 3,
        "draw_pile": [
            {"name": "Bash", "type": "attack"},
            {"name": "Strike", "type": "attack"},
            {"name": "Defend", "type": "skill"},
            {"name": "Defend", "type": "skill"},
        ],
        "discard_pile": [],
        "hand": [],
    }
    summary = draw_pile_feature_summary(player, _StubKB())
    assert summary["draw_pile_count"] == 4
    assert summary["attack_ratio_in_draw"] == 0.5
    assert summary["block_ratio_in_draw"] == 0.5
    assert summary["high_value_cards_in_draw"] == 1.0


def test_build_state_snapshot_includes_pile_features_in_combat() -> None:
    state = {
        "state_type": "monster",
        "player": {
            "hp": 50,
            "max_hp": 80,
            "block": 0,
            "energy": 3,
            "hand": [{"name": "Strike", "cost": 1, "type": "attack"}],
            "draw_pile": [
                {"name": "Bash", "type": "attack"},
                {"name": "Defend", "type": "skill"},
            ],
            "discard_pile": [],
            "relics": [],
            "potions": [],
        },
        "battle": {"enemies": [{"hp": 20, "max_hp": 20, "entity_id": "e0"}]},
    }
    snap = build_state_snapshot(state)
    assert snap["attack_ratio_in_draw"] == 0.5
    assert snap["expected_block_next_turn"] >= 0
    assert snap["expected_damage_next_turn"] >= 0

    vec = encode_snapshot(snap, state_type="monster")
    assert vec.shape == (FEATURE_DIM,)
    assert vec[8 : 8 + PILE_FEAT_DIM][0] == 0.5  # attack_ratio after player block
