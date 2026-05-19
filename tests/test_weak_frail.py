"""Step 1: Weak (enemy) and Frail (player) debuffs."""

from __future__ import annotations

from combat_sim.cards import defend, strike, uppercut
from combat_sim.damage_util import apply_frail_to_block, apply_weak_to_damage
from combat_sim.engine import CombatEngine
from combat_sim.state import CombatState, EnemyState, Intent, IntentKind


def test_apply_weak_formula() -> None:
    assert apply_weak_to_damage(8, 0) == 8
    assert apply_weak_to_damage(8, 1) == 6
    assert apply_weak_to_damage(13, 2) == 9


def test_apply_frail_formula() -> None:
    assert apply_frail_to_block(5, 0) == 5
    assert apply_frail_to_block(5, 1) == 3
    assert apply_frail_to_block(8, 1) == 6


def test_weak_reduces_enemy_attack_through_block() -> None:
    enemy = EnemyState(
        "e1",
        "Worm",
        hp=30,
        max_hp=30,
        weak_stacks=1,
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
    assert state.player_hp == 39  # floor(8*0.75)=6, 6 dmg − 5 block = 1


def test_frail_reduces_block_gained() -> None:
    state = CombatState.new_fight(
        deck=[defend()],
        player_hp=40,
        enemies=[EnemyState("e1", "Dummy", hp=99, max_hp=99)],
        seed=1,
    )
    state.frail_stacks = 1
    CombatEngine.play_card(state, state.hand[0].instance_id, None)
    assert state.player_block == 3


def test_uppercut_applies_weak_and_vuln() -> None:
    state = CombatState.new_fight(
        deck=[uppercut()],
        player_hp=40,
        enemies=[EnemyState("e1", "Target", hp=80, max_hp=80)],
        seed=0,
    )
    foe = state.enemies[0]
    card_id = state.hand[0].instance_id
    assert CombatEngine.play_card(state, card_id, "e1")
    assert foe.weak_stacks >= 1
    assert foe.vuln_stacks >= 1
    assert foe.hp < 80


def test_weak_decays_after_enemy_turn() -> None:
    enemy = EnemyState(
        "e1",
        "Worm",
        hp=30,
        max_hp=30,
        weak_stacks=2,
        pattern=[Intent(IntentKind.BLOCK, 5)],
    )
    state = CombatState(
        player_hp=40,
        player_max_hp=40,
        energy=3,
        enemies=[enemy],
    )
    CombatEngine.end_turn(state)
    assert enemy.weak_stacks == 1


def test_frail_decays_end_of_player_turn() -> None:
    state = CombatState.new_fight(
        deck=[strike()],
        player_hp=40,
        enemies=[EnemyState("e1", "Dummy", hp=99, max_hp=99)],
        seed=0,
    )
    state.frail_stacks = 2
    CombatEngine.end_turn(state)
    assert state.frail_stacks == 1
