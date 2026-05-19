"""Run Explorer dashboard helpers."""

from __future__ import annotations

import pandas as pd

from dashboard.run_explorer import (
    _chosen_map_index,
    _should_merge_fights,
    _solver_tags_from_text,
    _timeline_moments,
    card_picks_for_run,
    explorer_candidate_runs,
    macro_qwen_log_for_run,
    map_path_for_run,
    prepare_run_combat_view,
)


def test_explorer_sorts_by_floor() -> None:
    runs = pd.DataFrame(
        [
            {
                "run_id": "a",
                "agent_version": "ppo_v6",
                "source": "agent",
                "floors_reached": 3,
                "combat_summary": [{"won_fight": True}],
                "timestamp": "2026-01-01",
            },
            {
                "run_id": "b",
                "agent_version": "ppo_v6",
                "source": "agent",
                "floors_reached": 10,
                "combat_summary": [{"won_fight": True}],
                "timestamp": "2026-01-02",
            },
        ]
    )
    out = explorer_candidate_runs(runs, agent_version="ppo_v6")
    assert out.iloc[0]["run_id"] == "b"


def test_solver_tags_from_reasoning() -> None:
    text = "solver: lethal T1; enumerated 12 sequences; solver: executing cached plan"
    tags = _solver_tags_from_text(text)
    assert "lethal T1" in tags
    assert "cached plan" in tags


def test_macro_qwen_log_includes_prompt() -> None:
    decisions = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "state_type": "event",
                "floor": 4,
                "act": 1,
                "action": "choose_event_option",
                "action_index": 0,
                "action_reasoning": "qwen_macro: event",
                "qwen_macro": {
                    "state_type": "event",
                    "user_prompt": "Event: Neow\nChoose option",
                    "response": '{"action":"choose_event_option","index":0}',
                    "source": "qwen",
                    "parsed": {"reasoning": "Safe pick"},
                },
            }
        ]
    )
    log = macro_qwen_log_for_run(decisions, "r1")
    assert len(log) == 1
    assert bool(log.iloc[0]["has_qwen"])
    assert "Neow" in str(log.iloc[0]["prompt"])


def test_map_path_parses_options() -> None:
    decisions = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "state_type": "map",
                "floor": 2,
                "act": 1,
                "action": "choose_map_node",
                "action_index": 1,
                "action_reasoning": "option[0] Monster: 25. option[1] Rest: 40.",
            }
        ]
    )
    path = map_path_for_run(decisions, "r1")
    assert len(path) == 1
    assert path.iloc[0]["room"] == "Rest"
    assert "Monster" in path.iloc[0]["options"]


def test_timeline_flags_lost_fight() -> None:
    run = {
        "run_id": "r1",
        "won": False,
        "combat_summary": [
            {
                "enemy_names": ["Slime"],
                "won_fight": False,
                "damage_taken": 30,
                "hp_start": 50,
                "hp_end": 0,
                "state_type": "monster",
            }
        ],
        "cause_of_death": "monster combat - vs Slime",
    }
    moments = _timeline_moments(run, pd.DataFrame())
    assert any("Lost fight" in m["title"] for m in moments)
    assert any(m["severity"] == "critical" for m in moments)


def test_merge_bygone_effigy_segments() -> None:
    run = {
        "combat_summary": [
            {"enemy_names": ["Bygone Effigy"], "state_type": "elite", "turns": 5, "damage_taken": 10, "hp_start": 80, "hp_end": 60, "won_fight": True},
            {"enemy_names": ["Bygone Effigy"], "state_type": "elite", "turns": 3, "damage_taken": 20, "hp_start": 60, "hp_end": 30, "won_fight": True},
        ],
        "hp_before_each_combat": [80, 60],
        "hp_after_each_combat": [60, 30],
    }
    merged, _, _, _ = prepare_run_combat_view(run, pd.DataFrame())
    assert len(merged) == 1
    assert merged[0]["_merged_segments"] == 2
    assert merged[0]["damage_taken"] == 30


def test_policy_map_index_parsed() -> None:
    row = pd.Series(
        {
            "action_reasoning": "policy_net; key=choose_map_node:2 conf=99%",
            "map_choice_index": None,
        }
    )
    assert _chosen_map_index(row) == 2


def test_card_picks_from_decisions() -> None:
    decisions = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "state_type": "card_reward",
                "floor": 3,
                "act": 1,
                "card_reward_offered": ["Strike", "Defend", "Bash"],
                "card_reward_picked": "Bash",
                "action": "select_card_reward",
            }
        ]
    )
    picks = card_picks_for_run(decisions, "r1")
    assert picks.iloc[0]["picked"] == "Bash"
    assert "Bash" in picks.iloc[0]["offered"]
