"""Sim 1: Bash, Vulnerable, 10-card deck."""

from __future__ import annotations

from combat_sim.engine import CombatEngine
from combat_sim.scenarios import jaw_worm_sim1
from combat_sim.sim1.composition import TurnCompositionSim1, feasible_plays_sim1
from combat_sim.sim1.damage import damage_with_vulnerable
from combat_sim.sim1.opening import opening_compositions_sim1, opening_hand_probability
from combat_sim.runner import run_batch
from combat_sim.sim1.optimal import choose_optimal_turn_sim1, solve_optimal_sim1
from combat_sim.sim1.tuple_dp import _rank, apply_turn_sim1, make_opening_state, FightParams
from combat_sim.state import CombatPhase


def _assert_no_spurious_end_turn(
    decision,
    *,
    context: str = "",
) -> None:
    """End Turn must not win when a playable line is equal or better (memo/prune regression)."""
    prefix = f"{context}: " if context else ""
    end = TurnCompositionSim1(0, 0, 0)
    chosen_rank = _rank(decision.value)
    best_playable = max(
        (_rank(v) for c, v in decision.candidates if c.cards_played > 0),
        default=chosen_rank,
    )
    if best_playable >= chosen_rank and any(
        c.cards_played > 0 and _rank(v) >= chosen_rank for c, v in decision.candidates
    ):
        assert decision.composition != end, prefix + "chose End Turn despite playable lines"
        assert decision.composition.cards_played > 0, prefix + "empty composition"
    if decision.candidates:
        top = max(decision.candidates, key=lambda x: _rank(x[1]))
        assert _rank(decision.value) == _rank(top[1]), prefix + "chosen value not top candidate"


def test_vulnerable_damage_floor() -> None:
    assert damage_with_vulnerable(6, 1) == 9
    assert damage_with_vulnerable(8, 1) == 12
    assert damage_with_vulnerable(6, 0) == 6


def test_feasible_plays_energy() -> None:
    plays = feasible_plays_sim1(2, 2, 1)
    assert TurnCompositionSim1(1, 0, 1) in plays
    assert TurnCompositionSim1(3, 0, 0) not in plays
    assert all(p.energy_cost <= 3 for p in plays)


def test_opening_probs_sum_to_one() -> None:
    total = sum(p for *_, p in opening_compositions_sim1())
    assert abs(total - 1.0) < 1e-9


def test_bash_applies_vuln_after_damage() -> None:
    params = FightParams(80, 40, (("A", 8),))
    st = make_opening_state(params, hand_s=0, hand_d=0, hand_b=1, draw=())
    comp = TurnCompositionSim1(0, 0, 1)
    nxt = apply_turn_sim1(st, comp, params.pattern, shuffle_seed=0)
    assert nxt is not None
    assert nxt.hp_e == 40 - 8
    assert nxt.vuln_e == 1  # 2 applied, -1 at turn start


def test_sim1_logged_fight_wins() -> None:
    state = jaw_worm_sim1(seed=42)
    val = solve_optimal_sim1(state, shuffle_seed=42, max_turns=20)
    assert val.winnable
    decision = choose_optimal_turn_sim1(state, shuffle_seed=42, list_candidates=False)
    CombatEngine.apply_turn(state, decision.action)
    assert state.phase in (CombatPhase.PLAYER, CombatPhase.WON)


def test_turn1_seed42_picks_max_hp_line() -> None:
    """Regression: must not pick 1 Bash (73 HP) when 77 HP lines exist."""
    state = jaw_worm_sim1(seed=42)
    decision = choose_optimal_turn_sim1(state, shuffle_seed=42, list_candidates=True)
    assert decision.value.final_hp >= 77
    assert decision.composition != TurnCompositionSim1(0, 0, 1) or decision.value.final_hp >= 77
    top = max(c[1].final_hp for c in decision.candidates)
    assert decision.value.final_hp == top
    _assert_no_spurious_end_turn(decision, context="turn1")


def test_turn3_seed42_no_spurious_end_turn() -> None:
    """Regression: must play cards when winning lines exist (ceiling / sibling-bound bug)."""
    shuffle_seed = 42
    state = jaw_worm_sim1(seed=shuffle_seed)
    memo: dict = {}
    for _ in range(2):
        step = choose_optimal_turn_sim1(
            state,
            shuffle_seed=shuffle_seed,
            list_candidates=False,
            memo=memo,
        )
        CombatEngine.apply_turn(state, step.action)
    assert state.phase == CombatPhase.PLAYER
    assert state.turn == 3

    decision = choose_optimal_turn_sim1(
        state,
        shuffle_seed=shuffle_seed,
        list_candidates=True,
        memo=memo,
    )
    assert decision.value.winnable
    assert decision.value.final_hp >= 78
    _assert_no_spurious_end_turn(decision, context="turn3 seed42")


def test_sim1_jaw_worm_batch_100_seeds_all_win() -> None:
    report = run_batch(
        jaw_worm_sim1,
        scenario_name="jaw_worm_sim1",
        runs=100,
        solver="optimal",
        sim=1,
        max_turns=30,
        base_seed=0,
    )
    assert report.wins == 100, f"losses: {report.runs - report.wins}, hist={report.loss_turn_histogram}"
    assert report.win_rate == 1.0
