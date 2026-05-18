"""game_version tagging and training filters."""

from __future__ import annotations

import json
from pathlib import Path

from training.dataset import game_version_ok, load_decision_rows


def test_game_version_ok() -> None:
    assert game_version_ok("2026.05.18", "2026.05.01")
    assert not game_version_ok("2026.05.01", "2026.05.18")
    assert not game_version_ok(None, "2026.05.01")
    assert not game_version_ok("unknown", "2026.05.01")
    assert game_version_ok("2026.05.18", None)


def test_load_decision_rows_min_game_version(tmp_path: Path) -> None:
    runs = tmp_path / "runs.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    action = {"action": "end_turn"}
    rows_data = [
        {"run_id": "old", "run_score": 100.0, "game_version": "2026.05.01", "source": "agent"},
        {"run_id": "new", "run_score": 100.0, "game_version": "2026.05.18", "source": "agent"},
    ]
    runs.write_text("\n".join(json.dumps(r) for r in rows_data) + "\n", encoding="utf-8")
    decisions.write_text(
        "\n".join(
            json.dumps(
                {
                    "run_id": rid,
                    "game_version": gv,
                    "action_taken": action,
                    "state_snapshot": {"state_type": "monster"},
                }
            )
            for rid, gv in (("old", "2026.05.01"), ("new", "2026.05.18"))
        )
        + "\n",
        encoding="utf-8",
    )
    rows, scores = load_decision_rows(
        decisions,
        runs_path=runs,
        min_run_score_percentile=0.0,
        min_game_version="2026.05.18",
    )
    assert len(rows) == 1
    assert rows[0]["run_id"] == "new"
    assert "new" in scores
