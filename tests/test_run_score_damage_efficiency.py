"""Run score damage efficiency penalty."""

from sts2_agent.scorer import damage_efficiency_penalty, run_score


def test_damage_efficiency_penalty_missing_summary():
    assert damage_efficiency_penalty(None) == 0.0
    assert damage_efficiency_penalty([]) == 0.0


def test_damage_efficiency_penalty_calibration():
    summary = [{"turns": 10, "damage_dealt": 50}, {"turns": 10, "damage_dealt": 30}]
    # 80 damage / 20 turns = 4 dpt -> (10 - 4) * -12 = -72
    assert damage_efficiency_penalty(summary) == -72.0

    balanced = [{"turns": 10, "damage_dealt": 100}]
    assert damage_efficiency_penalty(balanced) == 0.0

    passive = [{"turns": 20, "damage_dealt": 140}]
    # 7 dpt -> (10 - 7) * -12 = -36
    assert damage_efficiency_penalty(passive) == -36.0


def test_run_score_includes_penalty():
    base = {
        "floors_reached": 10,
        "act_reached": 1,
        "avg_hp_pct_after_combat": 0.5,
        "bosses_killed": 0,
        "won": False,
    }
    without = run_score(base)
    with_penalty = run_score(
        {
            **base,
            "combat_summary": [{"turns": 10, "damage_dealt": 50}],
        }
    )
    assert with_penalty == without - 60.0  # 5 dpt -> -60
