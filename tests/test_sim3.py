"""Sim 3: Inflame, permanent Strength."""

from __future__ import annotations

from dataclasses import replace

from combat_sim.damage_util import damage_with_strength_and_vulnerable
from combat_sim.engine import CombatEngine
from combat_sim.runner import run_batch
from combat_sim.scenarios import BYGONE_EFFIGY_HP, bygone_effigy_sim3, jaw_worm_sim3
from combat_sim.sim1.damage import BASH_DAMAGE, STRIKE_DAMAGE
from combat_sim.sim3.composition import TurnCompositionSim3, feasible_plays_sim3
from combat_sim.sim3.optimal import choose_optimal_turn_sim3, solve_optimal_sim3
from combat_sim.sim3.tuple_dp import (
    INFLAME_STRENGTH,
    apply_turn_sim3,
    make_opening_state,
    FightParams,
)
from combat_sim.state import CombatPhase


def test_calc_damage_formula() -> None:
    from combat_sim.damage_util import calc_damage

    assert calc_damage(6, 0, 0, 0) == 6
    assert calc_damage(6, 2, 2, 3) == 15
    assert calc_damage(8, 2, 2, 3) == 19
    assert calc_damage(6, 2, 0, 1) == 8
    assert calc_damage(8, 2, 0, 1) == 11


def test_slow_play_order_bl_bash_two_strikes() -> None:
    """BL -> Bash -> Strike x2, Str 2, vuln after Bash."""
    from combat_sim.damage_util import calc_damage

    slow_n = 0
    slow_n += 1  # BL
    bash = calc_damage(8, 2, 0, slow_n)
    slow_n += 1
    vuln = 2
    s1 = calc_damage(6, 2, vuln, slow_n)
    slow_n += 1
    s2 = calc_damage(6, 2, vuln, slow_n)
    assert bash == 11
    assert s1 == 14
    assert s2 == 15
    assert bash + s1 + s2 == 40


def test_effigy_has_slow_and_resets_each_turn() -> None:
    from combat_sim.cards import strike
    from combat_sim.scenarios import _bygone_effigy_enemy
    from combat_sim.state import CombatState

    state = CombatState.new_fight(
        deck=[strike(), strike()],
        player_hp=80,
        enemies=[_bygone_effigy_enemy()],
        seed=1,
    )
    foe = state.enemies[0]
    assert foe.has_slow
    eid = foe.enemy_id
    c0, c1 = state.hand[0], state.hand[1]
    CombatEngine.play_card(state, c0.instance_id, eid)
    assert foe.slow_cards_this_turn == 1
    CombatEngine.play_card(state, c1.instance_id, eid)
    assert foe.hp == 127 - (6 + 6)
    CombatEngine.end_turn(state)
    assert foe.slow_cards_this_turn == 0


def test_effigy_dp_matches_engine_slow_turn1_bl_four_strikes() -> None:
    from combat_sim.sim3.composition import TurnCompositionSim3
    from combat_sim.sim3.tuple_dp import FightParams, apply_turn_sim3, make_opening_state

    params = FightParams(80, 127, (("A", 0), ("A", 0), ("A", 23)), enemy_slow=True)
    st = make_opening_state(
        params, hand_s=4, hand_d=0, hand_b=0, hand_bl=1, hand_inf=0, draw=()
    )
    comp = TurnCompositionSim3(4, 0, 0, 1, 0)
    nxt = apply_turn_sim3(st, comp, params.pattern, 0, enemy_slow=True)
    assert nxt is not None
    assert nxt.hp_e == 127 - 28


def test_damage_with_strength_and_vulnerable() -> None:
    assert damage_with_strength_and_vulnerable(6, 2, 0) == 8
    assert damage_with_strength_and_vulnerable(6, 2, 2) == 12
    assert damage_with_strength_and_vulnerable(8, 2, 2) == 15


def test_inflame_energy_constraint() -> None:
    plays = feasible_plays_sim3(1, 1, 0, 0, 1)
    assert TurnCompositionSim3(1, 0, 0, 0, 1) in plays
    assert TurnCompositionSim3(2, 0, 0, 0, 1) not in plays


def test_inflame_grants_strength_and_exhausts() -> None:
    params = FightParams(80, 40, (("B", 5),))
    st = make_opening_state(
        params, hand_s=0, hand_d=0, hand_b=0, hand_bl=0, hand_inf=1, draw=()
    )
    nxt = apply_turn_sim3(st, TurnCompositionSim3(0, 0, 0, 0, 1), params.pattern, 0)
    assert nxt is not None
    assert nxt.strength_p == INFLAME_STRENGTH
    assert nxt.hand_inf == 0
    assert "I" not in nxt.discard


def test_strike_scales_with_strength_after_inflame() -> None:
    params = FightParams(80, 40, (("B", 5),))
    st = make_opening_state(
        params, hand_s=1, hand_d=0, hand_b=0, hand_bl=0, hand_inf=0, draw=()
    )
    st = replace(st, strength_p=INFLAME_STRENGTH)
    nxt = apply_turn_sim3(st, TurnCompositionSim3(1, 0, 0, 0, 0), params.pattern, 0)
    assert nxt is not None
    assert nxt.hp_e == 40 - damage_with_strength_and_vulnerable(STRIKE_DAMAGE, 2, 0)


def test_sim3_jaw_worm_wins_seed42() -> None:
    state = jaw_worm_sim3(seed=42)
    val = solve_optimal_sim3(state, shuffle_seed=42, max_turns=20)
    assert val.winnable
    decision = choose_optimal_turn_sim3(state, shuffle_seed=42, list_candidates=False)
    CombatEngine.apply_turn(state, decision.action)
    assert state.phase in (CombatPhase.PLAYER, CombatPhase.WON)


def test_bygone_effigy_sim3_deck_size() -> None:
    state = bygone_effigy_sim3(seed=42)
    assert len(state.draw_pile) + len(state.hand) == 12


def test_engine_inflame_exhaust_and_strength() -> None:
    from combat_sim.cards import inflame, strike
    from combat_sim.state import CombatState, EnemyState, Intent, IntentKind

    enemy = EnemyState("e", "E", 20, 20, pattern=[Intent(IntentKind.ATTACK, 5)])
    state = CombatState.new_fight(
        deck=[inflame(), strike()],
        player_hp=80,
        enemies=[enemy],
        seed=1,
    )
    inf = next(c for c in state.hand if c.definition.card_id == "INFLAME")
    assert CombatEngine.play_card(state, inf.instance_id, None)
    assert state.player_strength == 2
    assert sum(1 for c in state.discard_pile if c.definition.card_id == "INFLAME") == 0
    strike_card = next(c for c in state.hand if c.definition.card_id == "STRIKE")
    assert CombatEngine.play_card(state, strike_card.instance_id, "e")
    assert state.enemies[0].hp == 20 - 8


def test_effigy_seed42_turn6_picks_winning_line() -> None:
    """Regression: survival prune must not cut states that can still kill in time."""
    from combat_sim.sim3.optimal import choose_optimal_turn_sim3

    state = bygone_effigy_sim3(seed=42)
    memo: dict = {}
    shuffle_seed = 42
    for _ in range(5):
        step = choose_optimal_turn_sim3(
            state, shuffle_seed=shuffle_seed, memo=memo, list_candidates=False
        )
        CombatEngine.apply_turn(state, step.action)
    step6 = choose_optimal_turn_sim3(
        state, shuffle_seed=shuffle_seed, memo=memo, list_candidates=True
    )
    assert step6.value.winnable
    assert step6.composition.cards_played > 0
    assert step6.composition.label() == step6.candidates[0][0].label()


def test_batch_effigy_sim3_runs() -> None:
    report = run_batch(
        bygone_effigy_sim3,
        scenario_name="Effigy Sim3",
        runs=3,
        solver="optimal",
        sim=3,
        base_seed=0,
        max_turns=30,
    )
    assert report.win_rate > 0 or report.avg_enemy_hp_remaining > 0
