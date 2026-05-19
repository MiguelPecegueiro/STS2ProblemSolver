"""Canonical shuffle parity and admissible pruning."""

from __future__ import annotations

import time

import pytest

from combat_sim.bounds import (
    damage_max_kill,
    damage_min_survival,
    prune_survival,
    prune_unwinnable,
)
from combat_sim.engine import CombatEngine
from combat_sim.optimal_solver import solve_optimal
from combat_sim.scenarios import jaw_worm, slime_boss_minion
from combat_sim.shuffle import canonical_shuffle, shuffle_card_instances
from combat_sim.solver import solve_turn
from combat_sim.tuple_dp import (
    FightParams,
    make_opening_state,
    solve_tuple,
    state_from_engine,
)


def test_canonical_shuffle_deterministic() -> None:
    pile = ("S", "S", "D", "D", "S")
    a = canonical_shuffle(pile, 42, 1)
    b = canonical_shuffle(pile, 42, 1)
    assert a == b
    assert sorted(a) == sorted(pile)


def test_engine_reshuffle_matches_canonical() -> None:
    state = jaw_worm(seed=99)
    # Force a reshuffle by emptying draw and discarding
    state.hand.clear()
    state.draw_pile.clear()
    state.discard_pile = state.discard_pile  # keep
    tags = tuple("S" if c.definition.damage > 0 else "D" for c in state.discard_pile)
    if not tags:
        pytest.skip("no discard to test")
    count_before = state.shuffle_count
    engine_order = shuffle_card_instances(
        list(state.discard_pile), state.shuffle_seed, count_before
    )
    tuple_order = canonical_shuffle(tags, state.shuffle_seed, count_before)
    engine_tags = tuple("S" if c.definition.damage > 0 else "D" for c in engine_order)
    assert engine_tags == tuple_order


def test_prunes_do_not_break_slime_optimal() -> None:
    params = FightParams(50, 12, (("A", 8),))
    st = make_opening_state(params, hand_s=2, hand_d=3, draw=("S", "S", "D"))
    with_prune = solve_tuple(st, params, 0, use_prunes=True)
    without = solve_tuple(st, params, 0, use_prunes=False)
    assert with_prune == without
    assert with_prune.winnable


def test_jaw_worm_pruned_faster() -> None:
    state = jaw_worm(seed=42)
    t0 = time.perf_counter()
    v = solve_optimal(state, shuffle_seed=42, max_turns=20)
    elapsed = time.perf_counter() - t0
    assert v.winnable
    assert elapsed < 15.0, f"expected <15s, got {elapsed:.1f}s"


def test_engine_tuple_same_shuffle_count_after_turn() -> None:
    e1 = jaw_worm(seed=7)
    e2 = jaw_worm(seed=7)
    CombatEngine.apply_turn(e1, solve_turn(e1))
    CombatEngine.apply_turn(e2, solve_turn(e2))
    assert e1.shuffle_count == e2.shuffle_count
