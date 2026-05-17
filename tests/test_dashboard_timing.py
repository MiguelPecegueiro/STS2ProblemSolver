"""Dashboard run duration helpers."""

from __future__ import annotations

import pandas as pd

from dashboard.app import (
    agent_decisions_by_version,
    card_pick_label_from_row,
    duration_from_decisions,
    enrich_runs_with_duration,
    format_duration,
    mean_run_duration_sec,
)


def test_format_duration() -> None:
    assert format_duration(45) == "45s"
    assert format_duration(125) == "2m 5s"
    assert format_duration(None) == "-"


def test_duration_from_decisions_and_enrich() -> None:
    decisions = pd.DataFrame(
        {
            "run_id": ["a", "a", "b"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 12:00:00+00:00",
                    "2026-01-01 12:10:00+00:00",
                    "2026-01-02 12:00:00+00:00",
                ],
                utc=True,
            ),
        }
    )
    spans = duration_from_decisions(decisions)
    assert float(spans["a"]) == 600.0

    runs = pd.DataFrame({"run_id": ["a", "b"], "agent_version": ["v1", "v1"]})
    enriched = enrich_runs_with_duration(runs, decisions)
    assert float(enriched.loc[enriched["run_id"] == "a", "run_duration_sec"].iloc[0]) == 600.0
    assert mean_run_duration_sec(enriched) == 600.0


def test_card_pick_label_from_policy_reasoning() -> None:
    row = pd.Series(
        {
            "action_reasoning": (
                "policy_net; policy_net class=73 key=select_card_reward:2 conf=100.0%"
            ),
            "card_index": 2,
        }
    )
    assert card_pick_label_from_row(row) == "reward slot 2"


def test_agent_decisions_by_version() -> None:
    runs = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "source": ["agent", "agent"],
            "agent_version": ["bc_v2", "ppo_v1"],
        }
    )
    decisions = pd.DataFrame(
        {
            "run_id": ["r1", "r1", "r2"],
            "state_type": ["monster", "card_reward", "monster"],
            "action": ["play_card", "select_card_reward", "end_turn"],
        }
    )
    groups = agent_decisions_by_version(decisions, runs)
    assert set(groups) == {"bc_v2", "ppo_v1"}
    assert len(groups["bc_v2"]) == 2
    assert len(groups["ppo_v1"]) == 1
