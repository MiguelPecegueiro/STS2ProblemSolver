"""Phase B (combat_summary) training filter."""

from __future__ import annotations

import json
from pathlib import Path

from training.dataset import load_decision_rows


def test_clean_only_keeps_human_and_phase_b_agent(tmp_path: Path) -> None:
    runs = tmp_path / "runs.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    action = {"action": "end_turn"}

    runs.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"run_id": "human1", "source": "human", "run_score": 50.0},
                {"run_id": "agent_old", "source": "agent", "run_score": 50.0},
                {
                    "run_id": "agent_b",
                    "source": "agent",
                    "run_score": 50.0,
                    "combat_summary": [{"turns": 3, "damage_dealt": 10}],
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    decisions.write_text(
        "\n".join(
            json.dumps(
                {
                    "run_id": rid,
                    "action_taken": action,
                    "state_snapshot": {"state_type": "monster"},
                }
            )
            for rid in ("human1", "agent_old", "agent_b", "agent_old")
        )
        + "\n",
        encoding="utf-8",
    )

    rows, _scores, meta = load_decision_rows(
        decisions,
        runs_path=runs,
        min_run_score_percentile=0.0,
        clean_only=True,
    )
    assert len(rows) == 2
    assert meta["rows_discarded_phase_b"] == 2
    assert meta["kept_human"] == 1
    assert meta["kept_agent_phase_b"] == 1
