"""Combat rules: play cards, end turn, enemy intents."""

from __future__ import annotations

from dataclasses import dataclass, field

from combat_sim.cards import CardDef
from combat_sim.damage_util import (
    apply_damage_to_enemy_hp_block,
    apply_frail_to_block,
    apply_weak_to_damage,
    calc_damage,
)
from combat_sim.pattern_util import parse_pattern_step
from combat_sim.state import CardInstance, CombatPhase, CombatState, EnemyState, IntentKind


@dataclass(frozen=True, slots=True)
class TurnAction:
    """One player turn: ordered card plays then end turn."""

    plays: tuple[tuple[int, str | None], ...] = ()
    # Each play is (card instance_id, enemy_id). enemy_id required for attacks.

    def labels(self, initial_hand: list[CardInstance]) -> list[str]:
        by_id = {c.instance_id: c.name for c in initial_hand}
        names = [by_id.get(iid, "?") for iid, _ in self.plays]
        names.append("End Turn")
        return names


@dataclass
class FightResult:
    won: bool
    turns: int
    player_hp: int
    actions: list[TurnAction] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


class CombatEngine:
    """Apply STS-like combat transitions on :class:`CombatState`."""

    @staticmethod
    def play_card(
        state: CombatState,
        card_instance_id: int,
        target_enemy_id: str | None,
    ) -> bool:
        if state.phase != CombatPhase.PLAYER:
            return False
        hand_index = _hand_index_for_instance(state, card_instance_id)
        if hand_index is None:
            return False

        card = state.hand[hand_index]
        if card.cost > state.energy:
            return False

        if card.definition.hp_loss > 0:
            state.player_hp = max(0, state.player_hp - card.definition.hp_loss)

        if card.definition.energy_gain > 0:
            state.energy += card.definition.energy_gain

        if card.definition.damage > 0:
            if not target_enemy_id:
                return False
            enemy = _enemy_by_id(state, target_enemy_id)
            if enemy is None or not enemy.is_alive():
                return False
            slow = enemy.slow_cards_this_turn if enemy.has_slow else 0
            dmg = calc_damage(
                card.definition.damage,
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
            if card.definition.vulnerable_apply > 0:
                enemy.vuln_stacks += card.definition.vulnerable_apply
            if card.definition.weak_apply > 0:
                enemy.weak_stacks += card.definition.weak_apply

        if card.definition.block > 0:
            gained = apply_frail_to_block(card.definition.block, state.frail_stacks)
            state.player_block += gained

        if card.definition.frail_apply > 0:
            state.frail_stacks += card.definition.frail_apply

        if card.definition.strength_apply > 0:
            state.player_strength += card.definition.strength_apply

        state.energy -= card.cost
        played = state.hand.pop(hand_index)
        if not card.definition.exhaust:
            state.discard_pile.append(played)
        for slow_enemy in state.living_enemies():
            if slow_enemy.has_slow:
                slow_enemy.slow_cards_this_turn += 1
        CombatEngine._check_outcome(state)
        return True

    @staticmethod
    def end_turn(state: CombatState) -> None:
        if state.phase != CombatPhase.PLAYER:
            return

        for enemy in state.enemies:
            enemy.slow_cards_this_turn = 0
            enemy.hp_lost_this_player_turn = 0

        state.frail_stacks = max(0, state.frail_stacks - 1)

        while state.hand:
            state.discard_pile.append(state.hand.pop())

        CombatEngine._enemy_turn(state)
        if state.phase != CombatPhase.PLAYER:
            return

        state.turn += 1
        state._begin_player_turn()
        CombatEngine._check_outcome(state)

    @staticmethod
    def apply_turn(state: CombatState, action: TurnAction) -> None:
        for instance_id, target_id in action.plays:
            if not CombatEngine.play_card(state, instance_id, target_id):
                break
        if state.phase == CombatPhase.PLAYER:
            CombatEngine.end_turn(state)

    @staticmethod
    def run_fight(
        state: CombatState,
        planner,
        *,
        max_turns: int = 30,
    ) -> FightResult:
        actions: list[TurnAction] = []
        log: list[str] = []

        while state.phase == CombatPhase.PLAYER and state.turn <= max_turns:
            if state.all_enemies_dead():
                state.phase = CombatPhase.WON
                break
            turn_action = planner(state)
            actions.append(turn_action)
            hand_snapshot = list(state.hand)
            log.append(f"T{state.turn}: " + ", ".join(turn_action.labels(hand_snapshot)))
            CombatEngine.apply_turn(state, turn_action)
            if state.player_hp <= 0:
                state.phase = CombatPhase.LOST
                break

        if state.phase == CombatPhase.PLAYER and state.turn > max_turns:
            log.append(f"Stopped after {max_turns} turns (no win/loss).")

        return FightResult(
            won=state.phase == CombatPhase.WON,
            turns=state.turn,
            player_hp=state.player_hp,
            actions=actions,
            log=log,
        )

    @staticmethod
    def _enemy_turn(state: CombatState) -> None:
        for enemy in state.living_enemies():
            intent = enemy.current_intent()
            if intent is None:
                continue
            if intent.kind == IntentKind.ATTACK:
                dmg = apply_weak_to_damage(intent.value, enemy.weak_stacks)
                CombatEngine._damage_player(state, dmg)
                if intent.enemy_block_bonus > 0:
                    enemy.block += intent.enemy_block_bonus
            elif intent.kind == IntentKind.BLOCK:
                enemy.block += intent.value
            enemy.pattern_index = (enemy.pattern_index + 1) % max(len(enemy.pattern), 1)
            enemy.weak_stacks = max(0, enemy.weak_stacks - 1)
            if state.player_hp <= 0:
                state.phase = CombatPhase.LOST
                return

    @staticmethod
    def _damage_player(state: CombatState, damage: int) -> None:
        remaining = damage
        if state.player_block > 0:
            absorbed = min(state.player_block, remaining)
            state.player_block -= absorbed
            remaining -= absorbed
        if remaining > 0:
            state.player_hp = max(0, state.player_hp - remaining)

    @staticmethod
    def _check_outcome(state: CombatState) -> None:
        if state.player_hp <= 0:
            state.phase = CombatPhase.LOST
        elif state.all_enemies_dead():
            state.phase = CombatPhase.WON


def _enemy_by_id(state: CombatState, enemy_id: str) -> EnemyState | None:
    for enemy in state.enemies:
        if enemy.enemy_id == enemy_id:
            return enemy
    return None


def _hand_index_for_instance(state: CombatState, instance_id: int) -> int | None:
    for idx, card in enumerate(state.hand):
        if card.instance_id == instance_id:
            return idx
    return None
