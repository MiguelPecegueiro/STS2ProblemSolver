"""General tuple state keyed by card_id."""

from __future__ import annotations

from dataclasses import dataclass

from combat_sim.card_effect import CardEffect
from combat_sim.general.hand import HandCounts, counts_from_tuple, counts_to_tuple
from combat_sim.general.pile import count_in_pile, remaining_in_deck
from combat_sim.sim3.tuple_dp import Pattern


@dataclass(frozen=True, slots=True)
class DeckContext:
    """Deck-wide constants for one fight."""

    effects: dict[str, CardEffect]
    base_energy: int = 3

    @classmethod
    def from_effects(cls, deck: tuple[CardEffect, ...], *, base_energy: int = 3) -> DeckContext:
        index: dict[str, CardEffect] = {}
        for eff in deck:
            index[eff.card_id] = eff
        return cls(effects=index, base_energy=base_energy)


@dataclass(frozen=True, slots=True)
class GeneralState:
    hp_p: int
    block_p: int = 0
    strength_p: int = 0
    frail_p: int = 0
    hp_e: int = 0
    block_e: int = 0
    vuln_e: int = 0
    weak_e: int = 0
    pattern_idx: int = 0
    hand: tuple[tuple[str, int], ...] = ()
    draw: tuple[str, ...] = ()
    discard: tuple[str, ...] = ()
    exhaust: tuple[str, ...] = ()
    powers: tuple[tuple[str, int], ...] = ()
    turn: int = 1
    shuffles: int = 0

    def hand_dict(self) -> HandCounts:
        return counts_from_tuple(self.hand)


def make_opening_state(
    *,
    player_hp: int,
    enemy_hp: int,
    hand: HandCounts,
    draw: tuple[str, ...],
) -> GeneralState:
    return GeneralState(
        hp_p=player_hp,
        hp_e=enemy_hp,
        hand=counts_to_tuple(hand),
        draw=draw,
    )


def state_from_engine(state, pattern: Pattern, ctx: DeckContext) -> GeneralState:
    from collections import Counter

    enemy = state.living_enemies()[0] if state.living_enemies() else state.enemies[0]
    hand_c = Counter(c.definition.card_id for c in state.hand)

    def pile(cards) -> tuple[str, ...]:
        return tuple(c.definition.card_id for c in cards)

    return GeneralState(
        hp_p=state.player_hp,
        block_p=state.player_block,
        strength_p=state.player_strength,
        frail_p=state.frail_stacks,
        hp_e=enemy.hp,
        block_e=enemy.block,
        vuln_e=enemy.vuln_stacks,
        weak_e=enemy.weak_stacks,
        pattern_idx=enemy.pattern_index % max(len(pattern), 1),
        hand=counts_to_tuple(dict(hand_c)),
        draw=pile(state.draw_pile),
        discard=pile(state.discard_pile),
        turn=state.turn,
        shuffles=state.shuffle_count,
    )


def card_remaining(st: GeneralState, card_id: str) -> int:
    return remaining_in_deck(st.hand, st.draw, st.discard, card_id)


def max_damage_card_in_deck(st: GeneralState, ctx: DeckContext) -> int:
    best = 0
    for cid, eff in ctx.effects.items():
        if eff.damage <= 0:
            continue
        if card_remaining(st, cid) > 0:
            best = max(best, eff.damage)
    return best
