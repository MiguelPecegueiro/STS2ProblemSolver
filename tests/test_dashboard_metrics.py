"""Unit tests for dashboard.metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from dashboard.metrics import (
    damage_mitigation_rate,
    early_version_warnings,
    human_tier_miss_rate,
    incoming_damage_from_snapshot,
    parse_death_enemy,
    parse_intent_damage_value,
    pick_rate_table,
    potion_hoard_death_rate,
    tier_rank,
)


def test_parse_death_enemy():
    cause = "elite combat vs Jaw Worm - hp reached 0"
    assert parse_death_enemy(cause) == "Jaw Worm"


def test_parse_intent_damage_value():
    assert parse_intent_damage_value(12) == 12
    assert parse_intent_damage_value("4x2") == 8
    assert parse_intent_damage_value("6X2") == 12
    assert parse_intent_damage_value("") == 0
    assert parse_intent_damage_value("4x4") == 16


def test_incoming_damage_from_snapshot():
    snap = {
        "enemies": [
            {"intent_value": 12},
            {"intent_value": "4x2"},
            {"intent_value": None},
        ]
    }
    assert incoming_damage_from_snapshot(snap) == 20


def test_early_warning_flags_underperformance():
    runs = pd.DataFrame(
        {
            "source": ["agent"] * 20,
            "agent_version": ["v1"] * 10 + ["v2"] * 10,
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC"),
            "floors_reached": [30] * 10 + [20] * 10,
        }
    )
    warnings = early_version_warnings(runs, window=10, threshold=0.15)
    assert any("v2" in w and "v1" in w for w in warnings)


def test_human_tier_miss():
    cc = pd.DataFrame(
        [
            {"offered": ["Strike", "Defend"], "picked": "Strike"},
        ]
    )
    # Depends on knowledge tiers; at minimum function runs without error
    rate = human_tier_miss_rate(cc)
    assert rate is None or 0 <= rate <= 100


def test_extract_card_pick_name_resolves_offered():
    import pandas as pd

    from dashboard.metrics import extract_card_pick_name

    row = pd.Series(
        {
            "action_reasoning": "policy_net; key=select_card_reward:2 conf=100%",
            "card_index": 2,
            "card_reward_offered": ["Strike", "Defend", "Bash"],
        }
    )
    assert extract_card_pick_name(row) == "Bash"


def test_normalize_card_name_matches_formats():
    from dashboard.metrics import normalize_card_name, normalize_pick_list, pick_rate_table

    assert normalize_card_name("Pommel Strike") == "POMMEL_STRIKE"
    assert normalize_card_name("POMMEL_STRIKE") == "POMMEL_STRIKE"
    assert normalize_card_name("pommel strike") == "POMMEL_STRIKE"
    assert normalize_card_name("reward slot 2") is None

    agent = normalize_pick_list(["Pommel Strike", "POMMEL_STRIKE"])
    assert agent == ["POMMEL_STRIKE", "POMMEL_STRIKE"]

    table = pick_rate_table(["Pommel Strike", "Pommel Strike"], ["POMMEL_STRIKE"])
    assert not table.empty
    assert len(table) == 1
    assert table.iloc[0]["Agent picks"] == 2
    assert table.iloc[0]["Human picks"] == 1


def test_pick_rate_table():
    table = pick_rate_table(["STRIKE", "STRIKE", "DEFEND"], ["STRIKE", "BASH"])
    assert not table.empty
    assert "Agent picks" in table.columns


def test_tier_rank():
    assert tier_rank("S") > tier_rank("B")


def test_aggregate_combat_by_enemy():
    import pandas as pd

    from dashboard.metrics import aggregate_combat_by_enemy, format_enemy_label

    assert format_enemy_label(["Slime", "Slime"]) == "Slime"
    assert format_enemy_label(["B", "A"]) == "A, B"

    runs = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "source": "agent",
                "combat_summary": [
                    {
                        "enemy_names": ["Jaw Worm"],
                        "turns": 10,
                        "damage_taken": 20,
                        "damage_dealt": 50,
                    },
                    {
                        "enemy_names": ["Jaw Worm"],
                        "turns": 14,
                        "damage_taken": 30,
                        "damage_dealt": 60,
                    },
                ],
            },
            {
                "run_id": "r2",
                "source": "agent",
                "combat_summary": [
                    {
                        "enemy_names": ["Cultist"],
                        "turns": 8,
                        "damage_taken": 5,
                        "damage_dealt": 40,
                    },
                ],
            },
        ]
    )
    agg = aggregate_combat_by_enemy(runs)
    jaw = agg[agg["enemy"] == "Jaw Worm"].iloc[0]
    assert jaw["fights"] == 2
    assert jaw["avg_turns"] == 12.0
    assert jaw["avg_damage_taken"] == 25.0


def test_filter_detail_phase_b():
    import pandas as pd

    from dashboard.metrics import filter_detail_phase_b, run_has_combat_summary

    assert run_has_combat_summary([{"turns": 1}])
    assert not run_has_combat_summary([])
    assert not run_has_combat_summary(None)

    runs = pd.DataFrame(
        [
            {"run_id": "a1", "source": "agent", "combat_summary": [{"turns": 1}]},
            {"run_id": "a2", "source": "agent", "combat_summary": []},
            {"run_id": "h1", "source": "human", "combat_summary": None},
        ]
    )
    decisions = pd.DataFrame(
        [
            {"run_id": "a1"},
            {"run_id": "a2"},
        ]
    )
    out_runs, out_dec, _, n_clean, n_total = filter_detail_phase_b(
        runs, decisions, pd.DataFrame()
    )
    assert n_clean == 1
    assert n_total == 2
    assert len(out_runs[out_runs["source"] == "agent"]) == 1
    assert len(out_runs[out_runs["source"] == "human"]) == 1
    assert len(out_dec) == 1


def test_damage_mitigation_rate():
    decisions = pd.DataFrame(
        [
            {
                "action": "play_card",
                "incoming_damage": 13,
                "hp_lost_this_turn": 0,
            },
            {
                "action": "end_turn",
                "incoming_damage": 13,
                "hp_lost_this_turn": 0,
            },
            {
                "action": "end_turn",
                "incoming_damage": 13,
                "hp_lost_this_turn": 3,
            },
            {
                "action": "end_turn",
                "incoming_damage": 13,
                "hp_lost_this_turn": 13,
            },
        ]
    )
    # Only end_turn rows: 100%, ~76.9%, 0% -> avg ~58.97%
    assert damage_mitigation_rate(decisions) == pytest.approx(100 * (1 + 10 / 13 + 0) / 3)


def test_potion_hoard_legacy_filled_only_list():
    runs = pd.DataFrame(
        {
            "source": ["agent", "agent"],
            "won": [False, False],
            "potions_at_death": [["p1"], []],
        }
    )
    rate = potion_hoard_death_rate(runs, default_max_slots=2)
    assert rate == 100.0


def test_potion_hoard_slot_aware_belt():
    runs = pd.DataFrame(
        {
            "source": ["agent", "agent", "agent"],
            "won": [False, False, False],
            "max_potion_slots": [3, 3, 3],
            "potions_at_death": [
                ["p1", None, "p3"],
                [None, None, None],
                ["a", "b", "c"],
            ],
        }
    )
    rate = potion_hoard_death_rate(runs)
    assert rate == pytest.approx(100.0 * 2 / 3)  # 2 hoard of 3


def test_potion_hoard():
    runs = pd.DataFrame(
        {
            "source": ["agent", "agent"],
            "won": [False, False],
            "max_potion_slots": [2, 2],
            "potions_at_death": [["p1", None], [None, None]],
        }
    )
    rate = potion_hoard_death_rate(runs)
    assert rate == 100.0
