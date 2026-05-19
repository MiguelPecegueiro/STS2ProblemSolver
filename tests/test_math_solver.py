"""Math solver: win probability, batch report, logged fight."""

from __future__ import annotations

import io
import sys

from combat_sim.math_solver import MathSolverConfig, choose_turn, estimate_win_probability
from combat_sim.runner import run_batch, run_logged_fight
from combat_sim.scenarios import jaw_worm, slime_boss_minion
from combat_sim.state import CombatPhase


def test_win_probability_terminal() -> None:
    state = jaw_worm(seed=1)
    state.phase = CombatPhase.WON
    assert estimate_win_probability(state, MathSolverConfig(estimate_rollouts=10)) == 1.0


def test_choose_turn_returns_composition() -> None:
    state = jaw_worm(seed=2)
    fast = MathSolverConfig(decision_rollouts=8, estimate_rollouts=8, seed=2)
    decision = choose_turn(state, fast)
    assert 0 <= decision.win_probability <= 1.0
    assert decision.composition.cards_played <= 3


def test_logged_fight_completes() -> None:
    fast = MathSolverConfig(decision_rollouts=6, estimate_rollouts=6, seed=3)
    result = run_logged_fight(
        slime_boss_minion,
        scenario_name="Slime",
        seed=3,
        solver="mc",
        config=fast,
        verbose=False,
    )
    assert result.turn_logs
    assert result.turn_logs[0].win_probability >= 0.0


def test_batch_report_win_rate() -> None:
    fast = MathSolverConfig(decision_rollouts=4, estimate_rollouts=4).fast_batch()
    report = run_batch(
        slime_boss_minion,
        scenario_name="Slime",
        runs=20,
        solver="mc",
        config=fast,
        fast=True,
        base_seed=10,
    )
    assert report.runs == 20
    assert 0.0 <= report.win_rate <= 1.0
    text = report.format_report()
    assert "Win rate" in text


def test_cli_batch_smoke(capsys) -> None:
    from combat_sim.__main__ import main

    old = sys.argv
    try:
        sys.argv = [
            "combat_sim",
            "--scenario",
            "slime",
            "--batch",
            "5",
            "--mc",
            "--fast",
            "--base-seed",
            "0",
        ]
        main()
    finally:
        sys.argv = old
    out = capsys.readouterr().out
    assert "Win rate" in out
