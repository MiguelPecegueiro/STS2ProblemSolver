"""Exact DP optimal solver."""

from __future__ import annotations

import pytest

from combat_sim.engine import CombatEngine
from combat_sim.optimal_solver import LOSS_HP, choose_optimal_turn, solve_optimal
from combat_sim.scenarios import jaw_worm, slime_boss_minion
from combat_sim.state import CombatPhase


def test_slime_lethal_turn_one() -> None:
    state = slime_boss_minion(hp=12, seed=0)
    val = solve_optimal(state, max_turns=15)
    assert val.winnable
    decision = choose_optimal_turn(state, max_turns=15, list_candidates=False)
    assert decision.value.winnable
    CombatEngine.apply_turn(state, decision.action)
    assert state.phase == CombatPhase.WON


@pytest.mark.slow
def test_dp_beats_greedy_shape_on_jaw_worm() -> None:
    # Deck order from seed=42; DP draw/shuffle continuation uses shuffle_seed=0 (default).
    state = jaw_worm(hp=40, seed=42)
    val = solve_optimal(state, max_turns=20, shuffle_seed=0)
    assert val.winnable
    assert val.final_hp == 78


def test_deterministic_same_seed() -> None:
    a = solve_optimal(slime_boss_minion(seed=7), max_turns=15)
    b = solve_optimal(slime_boss_minion(seed=7), max_turns=15)
    assert a == b


def test_logged_optimal_fight() -> None:
    from combat_sim.runner import run_logged_fight

    result = run_logged_fight(
        slime_boss_minion,
        scenario_name="Slime",
        seed=42,
        solver="optimal",
        verbose=False,
    )
    assert result.turn_logs
    assert result.turn_logs[0].win_probability in (0.0, 1.0)
