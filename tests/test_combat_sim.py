"""Minimal combat simulator: rules, scenarios, turn solver."""

from __future__ import annotations

from combat_sim.cards import IRONCLAD_STARTER_STRIKE_DEFEND, defend, strike
from combat_sim.engine import CombatEngine, TurnAction
from combat_sim.scenarios import jaw_worm, slime_boss_minion
from combat_sim.solver import solve_fight, solve_turn
from combat_sim.state import CombatPhase, CombatState, EnemyState, Intent, IntentKind


def test_strike_deals_damage_through_enemy_block() -> None:
    enemy = EnemyState("e1", "Slime", hp=20, max_hp=20, block=3)
    state = CombatState.new_fight(
        deck=[strike()],
        player_hp=40,
        enemies=[enemy],
        seed=1,
    )
    card_id = state.hand[0].instance_id
    assert CombatEngine.play_card(state, card_id, "e1")
    foe = state.enemies[0]
    assert foe.hp == 17  # 6 dmg, 3 block absorbed
    assert foe.block == 0


def test_defend_grants_block() -> None:
    state = CombatState.new_fight(
        deck=[defend()],
        player_hp=40,
        enemies=[EnemyState("e1", "Dummy", hp=99, max_hp=99)],
        seed=1,
    )
    card_id = state.hand[0].instance_id
    CombatEngine.play_card(state, card_id, None)
    assert state.player_block == 5


def test_player_block_absorbs_enemy_attack() -> None:
    enemy = EnemyState(
        "e1",
        "Slime",
        hp=30,
        max_hp=30,
        pattern=[Intent(IntentKind.ATTACK, 8)],
    )
    state = CombatState(
        player_hp=40,
        player_max_hp=40,
        player_block=5,
        energy=3,
        enemies=[enemy],
    )
    CombatEngine.end_turn(state)
    assert state.player_hp == 37  # 8 - 5 block


def test_lethal_two_strikes_one_turn() -> None:
    state = CombatState.new_fight(
        deck=[strike(), strike()],
        player_hp=40,
        enemies=[EnemyState("e1", "Slime", hp=10, max_hp=10)],
        seed=0,
    )
    action = solve_turn(state)
    CombatEngine.apply_turn(state, action)
    assert state.phase == CombatPhase.WON


def test_solve_turn_blocks_incoming_attack() -> None:
    state = slime_boss_minion(hp=30, seed=0)
    action = solve_turn(state)
    names = action.labels(list(state.hand))
    assert "Defend" in names or state.total_incoming_damage() == 0


def test_draw_pile_and_discard_cycle() -> None:
    state = CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_STRIKE_DEFEND),
        player_hp=80,
        enemies=[EnemyState("e1", "Dummy", hp=999, max_hp=999)],
        seed=99,
    )
    hand_size = len(state.hand)
    CombatEngine.end_turn(state)
    assert len(state.hand) == hand_size
    assert state.turn == 2


def test_jaw_worm_pattern_advances() -> None:
    enemy = EnemyState(
        "jw",
        "Jaw Worm",
        hp=40,
        max_hp=40,
        pattern=[
            Intent(IntentKind.ATTACK, 7),
            Intent(IntentKind.BLOCK, 5),
            Intent(IntentKind.ATTACK, 11),
        ],
    )
    assert enemy.current_intent() and enemy.current_intent().value == 7
    enemy.pattern_index = 1
    assert enemy.current_intent() and enemy.current_intent().kind == IntentKind.BLOCK


def test_solve_fight_beats_jaw_worm() -> None:
    state = jaw_worm(hp=40, seed=42)
    result = solve_fight(state, max_turns=20)
    assert result.won, result.log
    assert result.player_hp > 0


def test_empty_turn_end_is_legal() -> None:
    state = CombatState.new_fight(
        deck=[defend()],
        player_hp=40,
        enemies=[EnemyState("e1", "Slime", hp=5, max_hp=5)],
        seed=0,
    )
    CombatEngine.apply_turn(state, TurnAction())
    assert state.turn == 2
