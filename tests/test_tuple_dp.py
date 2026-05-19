"""Tuple DP core and opening expectation."""

from __future__ import annotations

import time

import pytest

from combat_sim.scenarios import jaw_worm, slime_boss_minion
from combat_sim.optimal_solver import fight_params_from_state, solve_optimal
from combat_sim.tuple_dp import (
    OPENING_TOTAL,
    exact_opening_expectation,
    opening_compositions,
    opening_hand_probability,
)


def test_opening_probabilities_sum_to_one() -> None:
    total = sum(p for _, _, p in opening_compositions())
    assert abs(total - 1.0) < 1e-9
    assert sum(opening_hand_probability(s, 5 - s) for s in range(6)) == pytest.approx(1.0)
    assert int(sum(opening_hand_probability(s, 5 - s) for s in range(6)) * OPENING_TOTAL) == OPENING_TOTAL


def test_slime_opening_expectation() -> None:
    from combat_sim.tuple_dp import FightParams

    report = exact_opening_expectation(
        FightParams(50, 12, (("A", 8),)),
        shuffle_seed=0,
        average_draw_orders=False,
    )
    assert 0.0 <= report.win_rate <= 1.0
    assert report.expected_final_hp > 0


def test_tuple_faster_than_baseline_slime() -> None:
    state = slime_boss_minion(seed=0)
    t0 = time.perf_counter()
    val = solve_optimal(state, shuffle_seed=0, max_turns=15)
    elapsed = time.perf_counter() - t0
    assert val.winnable
    assert elapsed < 5.0


@pytest.mark.slow
def test_jaw_worm_tuple_solve() -> None:
    state = jaw_worm(seed=42)
    t0 = time.perf_counter()
    val = solve_optimal(state, shuffle_seed=42, max_turns=20)
    elapsed = time.perf_counter() - t0
    assert val.winnable
    assert val.final_hp >= 79
    assert elapsed < 25.0, f"took {elapsed:.1f}s"
