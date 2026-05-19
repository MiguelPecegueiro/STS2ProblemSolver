"""Apply CardEffect to CombatState — general deterministic card engine (Phase 1)."""

from __future__ import annotations

from combat_sim.card_effect import CardEffect, CardTarget
from combat_sim.damage_util import (
    apply_damage_to_enemy_hp_block,
    apply_frail_to_block,
    calc_damage,
)
from combat_sim.engine import CombatEngine
from combat_sim.state import CombatPhase, CombatState, EnemyState


def _enemy_by_id(state: CombatState, enemy_id: str) -> EnemyState | None:
    for enemy in state.enemies:
        if enemy.enemy_id == enemy_id:
            return enemy
    return None


def _living_enemies(state: CombatState) -> list[EnemyState]:
    return [e for e in state.enemies if e.is_alive()]


def apply_card_effect(
    state: CombatState,
    effect: CardEffect,
    target_enemy_id: str | None,
) -> bool:
    """
    Apply one card's immediate effects. Mirrors CombatEngine.play_card semantics.
    Returns False if the play is illegal.
    """
    if state.phase != CombatPhase.PLAYER:
        return False
    if effect.cost > state.energy:
        return False

    if effect.damage > 0 and effect.target == CardTarget.ENEMY:
        if not target_enemy_id:
            return False
        enemy = _enemy_by_id(state, target_enemy_id)
        if enemy is None or not enemy.is_alive():
            return False

    if effect.hp_loss > 0:
        state.player_hp = max(0, state.player_hp - effect.hp_loss)

    if effect.energy_gain > 0:
        state.energy += effect.energy_gain

    targets: list[EnemyState] = []
    if effect.damage > 0:
        if effect.target == CardTarget.ALL_ENEMIES:
            targets = _living_enemies(state)
            if not targets:
                return False
        elif effect.target == CardTarget.ENEMY:
            enemy = _enemy_by_id(state, target_enemy_id or "")
            if enemy is None:
                return False
            targets = [enemy]

    for enemy in targets:
        for _ in range(max(1, effect.hits)):
            slow = enemy.slow_cards_this_turn if enemy.has_slow else 0
            dmg = calc_damage(
                effect.damage,
                state.player_strength,
                enemy.vuln_stacks,
                slow,
            )
            enemy.hp, enemy.block, hp_dealt = apply_damage_to_enemy_hp_block(
                enemy.hp,
                enemy.block,
                dmg,
                hp_loss_cap=enemy.hp_loss_cap_per_turn,
                hp_lost_this_turn=enemy.hp_lost_this_player_turn,
            )
            enemy.hp_lost_this_player_turn += hp_dealt
            if enemy.has_slow:
                enemy.slow_cards_this_turn += 1
        if effect.vuln_apply > 0:
            enemy.vuln_stacks += effect.vuln_apply
        if effect.weak_apply > 0:
            enemy.weak_stacks += effect.weak_apply

    if effect.block > 0:
        state.player_block += apply_frail_to_block(effect.block, state.frail_stacks)

    if effect.frail_apply > 0:
        state.frail_stacks += effect.frail_apply

    if effect.strength_apply > 0:
        state.player_strength += effect.strength_apply

    state.energy -= effect.cost

    # Non-attack cards still count for Slow (Bloodletting, Defend, Inflame, …).
    if effect.damage <= 0:
        for enemy in state.enemies:
            if enemy.has_slow:
                enemy.slow_cards_this_turn += 1

    CombatEngine._check_outcome(state)
    return True


def play_card_instance(
    state: CombatState,
    card_instance_id: int,
    effect: CardEffect,
    target_enemy_id: str | None,
) -> bool:
    """Play a specific hand card by instance id; discard/exhaust that instance."""
    if state.phase != CombatPhase.PLAYER:
        return False
    hand_index = None
    for idx, card in enumerate(state.hand):
        if card.instance_id == card_instance_id:
            hand_index = idx
            break
    if hand_index is None:
        return False
    if not apply_card_effect(state, effect, target_enemy_id):
        return False
    played = state.hand.pop(hand_index)
    if not effect.exhaust:
        state.discard_pile.append(played)
    return True
