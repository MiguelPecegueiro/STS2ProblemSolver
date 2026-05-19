"""Skulking Colony: Hardened Shell, attack cycle, Zoom block."""

from __future__ import annotations

from dataclasses import replace

from combat_sim.cards import strike
from combat_sim.damage_util import apply_damage_to_enemy_hp_block
from combat_sim.engine import CombatEngine
from combat_sim.scenarios import (
    SKULKING_COLONY_HP,
    SKULKING_COLONY_PATTERN,
    SKULKING_COLONY_PATTERN_TUPLE,
    skulking_colony_sim3,
    _skulking_colony_enemy,
)
from combat_sim.sim3.composition import TurnCompositionSim3
from combat_sim.sim3.optimal import choose_optimal_turn_sim3, solve_optimal_sim3
from combat_sim.sim3.tuple_dp import FightParams, _apply_forward_prunes, apply_turn_sim3, make_opening_state
from combat_sim.state import CombatState, IntentKind


def test_shell_aware_kill_bound() -> None:
    from combat_sim.damage_util import max_kill_total

    assert max_kill_total(50, 10, 15) == 150
    assert max_kill_total(50, 3, 15) == 45


def test_canonical_block_e_collapses_excess() -> None:
    from combat_sim.sim3.tuple_dp import canonical_block_e_for_key, max_single_hit_damage

    st = make_opening_state(
        FightParams(80, 70, SKULKING_COLONY_PATTERN_TUPLE, enemy_hp_loss_cap=15),
        hand_s=2,
        hand_d=0,
        hand_b=0,
        hand_bl=0,
        hand_inf=0,
        draw=(),
    )
    st = replace(st, block_e=20, strength_p=2, vuln_e=2)
    cap = max_single_hit_damage(st)
    assert cap == 12
    assert canonical_block_e_for_key(st) == 12
    st2 = replace(st, block_e=8)
    assert canonical_block_e_for_key(st2) == 8


def test_min_turns_to_kill_shell() -> None:
    from combat_sim.damage_util import min_turns_to_kill_shell, prune_kill_shell_turn_budget

    assert min_turns_to_kill_shell(70, 15) == 5
    assert min_turns_to_kill_shell(1, 15) == 1
    assert min_turns_to_kill_shell(16, 15) == 2
    assert not prune_kill_shell_turn_budget(70, 1, 25, 15)
    assert prune_kill_shell_turn_budget(70, 1, 4, 15)
    assert prune_kill_shell_turn_budget(70, 22, 25, 15)


def test_skulking_kill_prune_early_turn_budget() -> None:
    params = FightParams(
        80,
        70,
        SKULKING_COLONY_PATTERN_TUPLE,
        max_turns=4,
        enemy_hp_loss_cap=15,
    )
    st = make_opening_state(
        params, hand_s=0, hand_d=0, hand_b=0, hand_bl=0, hand_inf=0, draw=()
    )
    assert _apply_forward_prunes(st, params) == "kill"


def test_skulking_kill_prune_when_impossible_in_time() -> None:
    params = FightParams(
        80,
        70,
        (("A", 12), ("A", 14, 10), ("A", 9), ("A", 14)),
        max_turns=30,
        enemy_hp_loss_cap=15,
    )
    st = make_opening_state(
        params, hand_s=0, hand_d=0, hand_b=0, hand_bl=0, hand_inf=0, draw=()
    )
    st = replace(st, turn=28)
    assert _apply_forward_prunes(st, params) == "kill"


def test_hardened_shell_caps_hp_loss_per_turn() -> None:
    hp, block, dealt = apply_damage_to_enemy_hp_block(
        70, 0, 20, hp_loss_cap=15, hp_lost_this_turn=0
    )
    assert dealt == 15
    assert hp == 55
    hp2, block2, dealt2 = apply_damage_to_enemy_hp_block(
        hp, block, 20, hp_loss_cap=15, hp_lost_this_turn=dealt
    )
    assert dealt2 == 0
    assert hp2 == 55


def test_skulking_pattern_and_shell_on_enemy() -> None:
    enemy = _skulking_colony_enemy()
    assert enemy.hp_loss_cap_per_turn == 15
    assert len(enemy.pattern) == 4
    assert enemy.pattern[1].value == 14
    assert enemy.pattern[1].enemy_block_bonus == 10
    assert SKULKING_COLONY_PATTERN_TUPLE[1] == ("A", 14, 10)


def test_engine_shell_resets_on_end_turn() -> None:
    state = CombatState.new_fight(
        deck=[strike()],
        player_hp=80,
        enemies=[_skulking_colony_enemy()],
        seed=1,
    )
    foe = state.enemies[0]
    CombatEngine.play_card(state, state.hand[0].instance_id, foe.enemy_id)
    assert foe.hp_lost_this_player_turn == 6
    CombatEngine.end_turn(state)
    assert foe.hp_lost_this_player_turn == 0


def test_skulking_turn3_suboptimal_lines_not_false_loss() -> None:
    """Regression: ceiling prune must not mark sibling plays LOSS after a better line is found."""
    from combat_sim.sim3.tuple_dp import (
        LOSS_HP,
        _child_value,
        solve_with_best_play_sim3,
    )

    params = FightParams(
        80,
        55,
        SKULKING_COLONY_PATTERN_TUPLE,
        max_turns=25,
        enemy_hp_loss_cap=15,
    )
    st = make_opening_state(
        params, hand_s=1, hand_d=3, hand_b=0, hand_bl=1, hand_inf=0, draw=("D", "B", "S", "S", "S", "S")
    )
    st = replace(st, hp_p=61, hp_e=55, block_e=10, turn=3, pattern_idx=2)

    memo: dict = {}
    best_known = [LOSS_HP]
    solve_with_best_play_sim3(st, params, 42, memo=memo, best_known=best_known)
    assert best_known[0] > LOSS_HP

    comps = {
        TurnCompositionSim3(1, 3, 0, 1, 0): "bl_s_3d",
        TurnCompositionSim3(1, 2, 0, 0, 0): "s_2d",
        TurnCompositionSim3(0, 3, 0, 1, 0): "bl_3d",
    }
    for comp, _ in comps.items():
        val = _child_value(st, comp, params, 42, memo, [LOSS_HP], use_prunes=True)
        assert val.winnable, comp


def test_discard_order_invariant_same_multiset() -> None:
    """Same discard multiset after turn must not depend on play order (canonical sort)."""
    params = FightParams(
        80,
        55,
        SKULKING_COLONY_PATTERN_TUPLE,
        enemy_hp_loss_cap=15,
    )
    st = make_opening_state(
        params, hand_s=1, hand_d=3, hand_b=0, hand_bl=1, hand_inf=0, draw=("D", "B", "S", "S", "S", "S")
    )
    st = replace(st, hp_p=61, hp_e=55, block_e=10, turn=3, pattern_idx=2)
    bl_s_3d = apply_turn_sim3(
        st, TurnCompositionSim3(1, 3, 0, 1, 0), params.pattern, 42, enemy_hp_loss_cap=15
    )
    bl_3d = apply_turn_sim3(
        st, TurnCompositionSim3(0, 3, 0, 1, 0), params.pattern, 42, enemy_hp_loss_cap=15
    )
    assert bl_s_3d is not None and bl_3d is not None
    assert bl_s_3d.hp_p == bl_3d.hp_p == 58
    assert bl_s_3d.discard == bl_3d.discard == ("D", "D", "D", "L", "S")
    assert bl_s_3d.draw == bl_3d.draw
    assert bl_s_3d.hand_s == bl_3d.hand_s


def test_skulking_colony_sim3_solves() -> None:
    state = skulking_colony_sim3(seed=42)
    val = solve_optimal_sim3(state, shuffle_seed=42, max_turns=25)
    assert val.final_hp != 0
    assert state.enemies[0].max_hp == SKULKING_COLONY_HP


def test_skulking_dp_zoom_adds_enemy_block() -> None:
    from dataclasses import replace

    params = FightParams(
        80,
        70,
        SKULKING_COLONY_PATTERN_TUPLE,
        enemy_hp_loss_cap=15,
    )
    st = make_opening_state(
        params, hand_s=0, hand_d=0, hand_b=0, hand_bl=0, hand_inf=0, draw=()
    )
    st = replace(st, pattern_idx=1, hp_p=50)
    nxt = apply_turn_sim3(
        st,
        TurnCompositionSim3(0, 0, 0, 0, 0),
        params.pattern,
        0,
        enemy_hp_loss_cap=15,
    )
    assert nxt is not None
    assert nxt.block_e >= 10
