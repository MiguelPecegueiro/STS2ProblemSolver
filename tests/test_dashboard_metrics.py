"""Unit tests for dashboard.metrics."""

from __future__ import annotations

import pandas as pd

from dashboard.metrics import (
    early_version_warnings,
    human_tier_miss_rate,
    incoming_damage_from_snapshot,
    parse_death_enemy,
    pick_rate_table,
    potion_hoard_death_rate,
    tier_rank,
)


def test_parse_death_enemy():
    cause = "elite combat vs Jaw Worm - hp reached 0"
    assert parse_death_enemy(cause) == "Jaw Worm"


def test_incoming_damage_from_snapshot():
    snap = {
        "enemies": [
            {"intent_value": 12},
            {"intent_value": 5},
            {"intent_value": None},
        ]
    }
    assert incoming_damage_from_snapshot(snap) == 17


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


def test_pick_rate_table():
    table = pick_rate_table(["A", "A", "B"], ["A", "C"])
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


def test_block_efficiency_uses_flattened_block_applied():
    import pandas as pd

    from dashboard.metrics import block_efficiency

    decisions = pd.DataFrame(
        [
            {
                "state_type": "monster",
                "incoming_damage": 10,
                "block_applied": 5,
                "immediate_reward": None,
            },
            {
                "state_type": "monster",
                "incoming_damage": 8,
                "block_applied": 0,
                "immediate_reward": None,
            },
            {
                "state_type": "monster",
                "incoming_damage": 0,
                "block_applied": 9,
                "immediate_reward": None,
            },
        ]
    )
    assert block_efficiency(decisions) == 50.0


def test_potion_hoard():
    runs = pd.DataFrame(
        {
            "source": ["agent", "agent"],
            "won": [False, False],
            "potions_at_death": [["p1"], []],
        }
    )
    rate = potion_hoard_death_rate(runs, max_slots=2)
    assert rate == 100.0
