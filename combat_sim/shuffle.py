"""Canonical shuffle — single source of truth for engine and tuple DP."""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")

# Stable multiset order before shuffle (play order must not affect draw RNG).
_TAG_RANK = {"B": 0, "D": 1, "I": 2, "L": 3, "S": 4, "T": 5}


def sort_tag_pile(pile: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(pile, key=lambda t: _TAG_RANK.get(t, 99)))


def canonical_shuffle(
    pile: tuple[T, ...] | list[T],
    seed: int,
    shuffle_count: int,
) -> tuple[T, ...]:
    """Deterministic shuffle: shuffle(pile, seed, shuffle_count) -> ordered pile."""
    rng = random.Random(seed + shuffle_count)
    cards = list(pile)
    rng.shuffle(cards)
    return tuple(cards)


def card_tag(definition) -> str:
    cid = str(definition.card_id or "").upper()
    if cid == "BASH":
        return "B"
    if cid == "BLOODLETTING":
        return "L"
    if cid == "INFLAME":
        return "I"
    if cid == "TWIN_STRIKE":
        return "T"
    if definition.damage > 0:
        return "S"
    return "D"


def shuffle_card_instances(
    cards: list,
    seed: int,
    shuffle_count: int,
) -> list:
    """Shuffle a list of CardInstance using canonical tags."""
    if not cards:
        return []
    tagged = [(card_tag(c.definition), c) for c in cards]
    tag_tuple = sort_tag_pile(tuple(t for t, _ in tagged))
    order = canonical_shuffle(tag_tuple, seed, shuffle_count)
    pool = list(cards)
    out: list = []
    for want in order:
        for i, card in enumerate(pool):
            if card_tag(card.definition) == want:
                out.append(pool.pop(i))
                break
    return out
