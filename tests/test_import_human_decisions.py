"""Tests for tools/import_human_decisions.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.import_human_decisions import (  # noqa: E402
    card_choice_to_decision,
    import_human_decisions,
    load_existing_keys,
)


def test_card_choice_to_decision_select() -> None:
    row = {
        "run_id": "r1",
        "source": "human",
        "floor": 5,
        "act": 1,
        "offered": ["STRIKE", "DEFEND", "BASH"],
        "picked": "BASH",
        "hp_at_pick": 60,
    }
    dec = card_choice_to_decision(row)
    assert dec is not None
    assert dec["state_type"] == "card_reward"
    assert dec["action_taken"] == {"action": "select_card_reward", "card_index": 2}
    assert dec["source"] == "human"
    assert dec["agent_version"] == "human"
    assert dec["immediate_reward"] == 0
    assert dec["card_reward_offered"] == ["STRIKE", "DEFEND", "BASH"]


def test_card_choice_to_decision_skip() -> None:
    row = {
        "run_id": "r1",
        "source": "human",
        "floor": 6,
        "act": 1,
        "offered": ["STRIKE", "DEFEND"],
        "picked": None,
    }
    dec = card_choice_to_decision(row)
    assert dec is not None
    assert dec["action_taken"] == {"action": "skip_card_reward"}


def test_import_dedupes_by_run_id_and_floor(tmp_path: Path) -> None:
    choices = tmp_path / "card_choices.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    choices.write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "source": "human",
                "floor": 1,
                "act": 1,
                "offered": ["A", "B"],
                "picked": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    stats1 = import_human_decisions(
        card_choices_path=choices,
        decisions_path=decisions,
    )
    assert stats1["added_total"] == 1

    stats2 = import_human_decisions(
        card_choices_path=choices,
        decisions_path=decisions,
    )
    assert stats2["added_total"] == 0
    assert stats2["skipped_duplicate"] == 1

    keys = load_existing_keys(decisions)
    assert ("run-a", 1) in keys
