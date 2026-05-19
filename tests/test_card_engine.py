"""General card engine (CardEffect) matches legacy CombatEngine."""

from __future__ import annotations

from combat_sim.card_effect import (
    BASH_EFFECT,
    BLOODLETTING_EFFECT,
    DEFEND_EFFECT,
    STRIKE_EFFECT,
    TWIN_STRIKE_EFFECT,
    UPPERCUT_EFFECT,
    card_effect_from_card_def,
)
from combat_sim.card_engine import apply_card_effect, play_card_instance
from combat_sim.cards import bash, bloodletting, defend, strike, uppercut
from combat_sim.engine import CombatEngine
from combat_sim.state import CombatPhase, CombatState, EnemyState


def test_strike_via_card_effect_matches_engine() -> None:
    enemy_legacy = EnemyState("e1", "Slime", hp=20, max_hp=20, block=3)
    state_legacy = CombatState.new_fight(deck=[strike()], player_hp=40, enemies=[enemy_legacy], seed=1)
    CombatEngine.play_card(state_legacy, state_legacy.hand[0].instance_id, "e1")

    enemy = EnemyState("e1", "Slime", hp=20, max_hp=20, block=3)
    state = CombatState.new_fight(deck=[strike()], player_hp=40, enemies=[enemy], seed=1)
    play_card_instance(state, state.hand[0].instance_id, STRIKE_EFFECT, "e1")

    assert state.enemies[0].hp == state_legacy.enemies[0].hp
    assert state.energy == state_legacy.energy


def test_twin_strike_two_hits() -> None:
    state = CombatState.new_fight(
        deck=[],
        player_hp=40,
        enemies=[EnemyState("e1", "Slime", hp=20, max_hp=20)],
        seed=0,
    )
    state.energy = 3
    assert apply_card_effect(state, TWIN_STRIKE_EFFECT, "e1")
    assert state.enemies[0].hp == 10  # 5 + 5


def test_card_def_bridge() -> None:
    eff = card_effect_from_card_def(uppercut())
    assert eff.damage == 13
    assert eff.weak_apply == 1
    assert eff.vuln_apply == 1


def test_bloodletting_energy_via_effect() -> None:
    state = CombatState.new_fight(
        deck=[bloodletting()],
        player_hp=40,
        enemies=[EnemyState("e1", "X", hp=99, max_hp=99)],
        seed=0,
    )
    eff = card_effect_from_card_def(bloodletting())
    play_card_instance(state, state.hand[0].instance_id, eff, None)
    assert state.player_hp == 37
    assert state.energy == 5  # 3 + 2


def test_defend_block_with_frail() -> None:
    state = CombatState.new_fight(
        deck=[defend()],
        player_hp=40,
        enemies=[EnemyState("e1", "X", hp=99, max_hp=99)],
        seed=0,
    )
    state.frail_stacks = 1
    play_card_instance(state, state.hand[0].instance_id, DEFEND_EFFECT, None)
    assert state.player_block == 3


def test_bash_vuln_via_effect() -> None:
    state = CombatState.new_fight(
        deck=[bash()],
        player_hp=40,
        enemies=[EnemyState("e1", "X", hp=80, max_hp=80)],
        seed=0,
    )
    play_card_instance(state, state.hand[0].instance_id, BASH_EFFECT, "e1")
    assert state.enemies[0].vuln_stacks == 2
    assert state.enemies[0].hp == 72
