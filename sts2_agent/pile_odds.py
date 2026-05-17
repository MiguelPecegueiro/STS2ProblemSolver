"""Draw/discard pile analysis for next-turn combat estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sts2_agent.knowledge import KnowledgeBase
from sts2_agent.scorer import (
    _card_block_value,
    _card_cost,
    _card_damage_value,
    _hand_card_is_attack,
    _hand_card_is_block,
    card_name,
)


# Normalized id substrings - any draw-pile card whose name/id contains one counts as high-value.
HIGH_VALUE_DRAW_CARD_KEYS = frozenset(
    {
        "bash",
        "offering",
        "corpse_explosion",
        "demon_form",
        "feed",
        "meteor_strike",
        "immolate",
        "grand_finale",
        "wish",
        "panache",
        "phantasmal_killer",
        "bludgeon",
        "flame_barrier",
        "ghostly_armor",
    }
)


def _normalize_card_key(text: str) -> str:
    return str(text or "").lower().replace(" ", "_")


def _card_is_high_value(card: dict) -> bool:
    name = _normalize_card_key(card_name(card))
    cid = _normalize_card_key(str(card.get("id") or ""))
    for key in HIGH_VALUE_DRAW_CARD_KEYS:
        if key in name or key in cid:
            return True
    return False


def draw_pile_feature_summary(
    player: dict,
    kb: KnowledgeBase,
    *,
    draw_count: int = 5,
    energy: int | None = None,
) -> dict[str, float | int]:
    """Pile composition + next-turn estimates for policy snapshots."""
    draw = _pile_cards(player, "draw_pile")
    n_draw = len(draw)

    attack_cards = 0
    block_cards = 0
    high_value_cards = 0
    for card in draw:
        codex = kb.lookup_card(card_name(card))
        if _hand_card_is_attack(card, codex):
            attack_cards += 1
        if _hand_card_is_block(card, codex):
            block_cards += 1
        if _card_is_high_value(card):
            high_value_cards += 1

    attack_ratio = float(attack_cards) / n_draw if n_draw else 0.0
    block_ratio = float(block_cards) / n_draw if n_draw else 0.0

    est = next_turn_combat_estimates(
        player, kb, draw_count=draw_count, energy=energy
    )

    return {
        "draw_pile_count": n_draw,
        "attack_cards_in_draw": attack_cards,
        "block_cards_in_draw": block_cards,
        "attack_ratio_in_draw": attack_ratio,
        "block_ratio_in_draw": block_ratio,
        "high_value_cards_in_draw": float(min(high_value_cards, 1)),
        "expected_block_next_turn": int(est.expected_block),
        "expected_damage_next_turn": int(est.expected_damage),
    }


@dataclass
class NextTurnEstimates:
    draw_count: int = 5
    deterministic: bool = False
    expected_block: int = 0
    expected_damage: int = 0
    prob_any_block: float = 0.0
    prob_any_attack: float = 0.0
    known_next_cards: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _pile_cards(player: dict, key: str) -> list[dict]:
    pile = player.get(key)
    if isinstance(pile, list):
        return [c for c in pile if isinstance(c, dict)]
    return []


def _hypergeom_at_least_one(successes: int, population: int, draws: int) -> float:
    """P(draw >= 1 success) when drawing `draws` cards without replacement."""
    if successes <= 0 or population <= 0 or draws <= 0:
        return 0.0
    if draws >= population:
        return 1.0 if successes > 0 else 0.0
    fails = population - successes
    if draws > fails:
        return 1.0
    # 1 - C(fails, draws) / C(population, draws)
    return 1.0 - math.comb(fails, draws) / math.comb(population, draws)


def _greedy_pile_value(
    cards: list[dict],
    energy: int,
    kb: KnowledgeBase,
    *,
    block: bool,
) -> int:
    remaining = energy
    options: list[tuple[int, int]] = []
    for card in cards:
        codex = kb.lookup_card(card_name(card))
        cost = _card_cost(card)
        if cost > energy:
            continue
        if block and _hand_card_is_block(card, codex):
            options.append((cost, _card_block_value(card, codex)))
        elif not block and _hand_card_is_attack(card, codex):
            options.append((cost, int(_card_damage_value(card, codex))))
    options.sort(key=lambda x: (-x[1], x[0]))
    total = 0
    for cost, value in options:
        if cost <= remaining:
            total += value
            remaining -= cost
    return total


def next_turn_combat_estimates(
    player: dict,
    kb: KnowledgeBase,
    *,
    draw_count: int = 5,
    energy: int | None = None,
) -> NextTurnEstimates:
    """Estimate block/damage available on the next turn's opening hand."""
    hand = _pile_cards(player, "hand")
    draw = _pile_cards(player, "draw_pile")
    discard = _pile_cards(player, "discard_pile")

    if energy is None:
        energy = int(player.get("max_energy") or player.get("energy") or 3)

    est = NextTurnEstimates(draw_count=draw_count)
    est.reasons.append(
        f"piles: draw={len(draw)} discard={len(discard)} hand→discard={len(hand)}"
    )

    # End of turn: hand shuffled into discard; draw from top of draw pile.
    if len(draw) >= draw_count:
        next_cards = draw[:draw_count]
        est.deterministic = True
        est.known_next_cards = [card_name(c) for c in next_cards]
        est.expected_block = _greedy_pile_value(next_cards, energy, kb, block=True)
        est.expected_damage = _greedy_pile_value(next_cards, energy, kb, block=False)
        est.prob_any_block = (
            1.0
            if any(
                _hand_card_is_block(c, kb.lookup_card(card_name(c))) for c in next_cards
            )
            else 0.0
        )
        est.prob_any_attack = (
            1.0
            if any(
                _hand_card_is_attack(c, kb.lookup_card(card_name(c))) for c in next_cards
            )
            else 0.0
        )
        est.reasons.append(
            f"deterministic next hand: {est.known_next_cards[:draw_count]}"
        )
        est.reasons.append(
            f"est next-turn block={est.expected_block} damage={est.expected_damage}"
        )
        return est

    # Draw pile exhausted partially - pool is draw + discard + hand (post-turn).
    pool = draw + discard + hand
    if not pool:
        est.reasons.append("empty deck - no next-turn estimate")
        return est

    block_cards = 0
    attack_cards = 0
    for card in pool:
        codex = kb.lookup_card(card_name(card))
        if _hand_card_is_block(card, codex):
            block_cards += 1
        if _hand_card_is_attack(card, codex):
            attack_cards += 1

    est.prob_any_block = _hypergeom_at_least_one(block_cards, len(pool), draw_count)
    est.prob_any_attack = _hypergeom_at_least_one(attack_cards, len(pool), draw_count)

    avg_block = (
        sum(
            _card_block_value(c, kb.lookup_card(card_name(c)))
            for c in pool
            if _hand_card_is_block(c, kb.lookup_card(card_name(c)))
        )
        / max(block_cards, 1)
    )
    avg_damage = (
        sum(
            _card_damage_value(c, kb.lookup_card(card_name(c)))
            for c in pool
            if _hand_card_is_attack(c, kb.lookup_card(card_name(c)))
        )
        / max(attack_cards, 1)
    )

    # Expected playable cards drawn (without replacement, rough).
    exp_block_cards = draw_count * (block_cards / len(pool))
    exp_attack_cards = draw_count * (attack_cards / len(pool))
    est.expected_block = int(min(exp_block_cards, block_cards) * avg_block)
    est.expected_damage = int(min(exp_attack_cards, attack_cards) * avg_damage)

    est.reasons.append(
        f"prob next hand block={est.prob_any_block:.0%} attack={est.prob_any_attack:.0%}"
    )
    est.reasons.append(
        f"est next-turn block≈{est.expected_block} damage≈{est.expected_damage}"
    )
    return est


def format_pile_summary(player: dict) -> dict[str, Any]:
    """Compact summary of draw/discard contents for logging."""
    draw = _pile_cards(player, "draw_pile")
    discard = _pile_cards(player, "discard_pile")
    return {
        "draw_count": len(draw),
        "discard_count": len(discard),
        "draw_top": [card_name(c) for c in draw[:8]],
        "discard_top": [card_name(c) for c in discard[-8:]],
    }
