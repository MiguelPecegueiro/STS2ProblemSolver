"""Sim 2: Bloodletting, dynamic energy."""

from __future__ import annotations

from dataclasses import replace

from combat_sim.engine import CombatEngine
from combat_sim.runner import run_batch
from combat_sim.scenarios import BYGONE_EFFIGY_HP, bygone_effigy_sim2, jaw_worm_sim2
from combat_sim.sim2.composition import TurnCompositionSim2, feasible_plays_sim2
from combat_sim.sim2.optimal import choose_optimal_turn_sim2, solve_optimal_sim2
from combat_sim.sim2.tuple_dp import (
    BLOODLETTING_HP_LOSS,
    apply_turn_sim2,
    make_opening_state,
    FightParams,
)
from combat_sim.state import CombatPhase


def test_bloodletting_energy_constraint() -> None:
    plays = feasible_plays_sim2(2, 1, 1, 1)
    assert TurnCompositionSim2(2, 0, 1, 1) in plays  # BL + 2 strike + bash = 1+2+2=5
    assert TurnCompositionSim2(3, 0, 1, 0) not in plays  # 3+2=5 > 3 without BL


def test_bloodletting_hp_loss_in_dp() -> None:
    params = FightParams(80, 40, (("B", 5),))  # block intent — no player damage this turn
    st = make_opening_state(params, hand_s=0, hand_d=0, hand_b=0, hand_bl=1, draw=())
    comp = TurnCompositionSim2(0, 0, 0, 1)
    nxt = apply_turn_sim2(st, comp, params.pattern, shuffle_seed=0)
    assert nxt is not None
    assert nxt.hp_p == 80 - BLOODLETTING_HP_LOSS


def test_bloodletting_lethal_returns_none() -> None:
    params = FightParams(80, 40, (("B", 5),))
    st = make_opening_state(params, hand_s=0, hand_d=0, hand_b=0, hand_bl=1, draw=())
    st = replace(st, hp_p=3)
    nxt = apply_turn_sim2(st, TurnCompositionSim2(0, 0, 0, 1), params.pattern, 0)
    assert nxt is None


def test_sim2_jaw_worm_wins_seed42() -> None:
    state = jaw_worm_sim2(seed=42)
    val = solve_optimal_sim2(state, shuffle_seed=42, max_turns=20)
    assert val.winnable
    decision = choose_optimal_turn_sim2(state, shuffle_seed=42, list_candidates=False)
    CombatEngine.apply_turn(state, decision.action)
    assert state.phase in (CombatPhase.PLAYER, CombatPhase.WON)


def test_no_spurious_end_turn_when_plays_win() -> None:
    """Regression pattern from Sim 1 memo bug."""
    shuffle_seed = 42
    state = jaw_worm_sim2(seed=shuffle_seed)
    memo: dict = {}
    step = choose_optimal_turn_sim2(
        state, shuffle_seed=shuffle_seed, list_candidates=True, memo=memo
    )
    assert step.value.winnable
    top = max(step.candidates, key=lambda x: (x[1].winnable, x[1].final_hp))
    assert step.composition.cards_played > 0
    assert step.value.final_hp == top[1].final_hp


def test_bygone_effigy_scenario_setup() -> None:
    state = bygone_effigy_sim2(seed=42)
    enemy = state.enemies[0]
    assert enemy.name == "Bygone Effigy"
    assert enemy.max_hp == BYGONE_EFFIGY_HP
    assert len(enemy.pattern) == 3
    assert enemy.pattern[0].value == 0
    assert enemy.pattern[2].value == 23
    assert len(state.draw_pile) + len(state.hand) == 11


def test_batch_reports_avg_enemy_hp_remaining() -> None:
    report = run_batch(
        bygone_effigy_sim2,
        scenario_name="Effigy",
        runs=5,
        solver="optimal",
        sim=2,
        base_seed=42,
        max_turns=30,
    )
    assert "Avg enemy HP left" in report.format_report()
    assert report.win_rate > 0 or report.avg_enemy_hp_remaining > 0


def test_bygone_effigy_optimal_solve_runs() -> None:
    """Tier A elite — solver completes (win/loss both valid)."""
    state = bygone_effigy_sim2(seed=42)
    val = solve_optimal_sim2(state, shuffle_seed=42, max_turns=40)
    assert val.final_hp != 0


def test_engine_bloodletting_grants_energy() -> None:
    from combat_sim.cards import bloodletting, strike
    from combat_sim.state import CombatState, EnemyState, Intent, IntentKind

    enemy = EnemyState("e", "E", 20, 20, pattern=[Intent(IntentKind.ATTACK, 5)])
    state = CombatState.new_fight(
        deck=[bloodletting(), strike()],
        player_hp=80,
        enemies=[enemy],
        seed=1,
    )
    bl = next(c for c in state.hand if c.card_id == "BLOODLETTING")
    assert CombatEngine.play_card(state, bl.instance_id, None)
    assert state.player_hp == 77
    assert state.energy == 5  # 3 - 0 + 2
