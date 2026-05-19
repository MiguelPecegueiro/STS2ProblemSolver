"""Draw/discard piles as ordered card_id tuples."""

from __future__ import annotations

from combat_sim.shuffle import canonical_shuffle, sort_tag_pile

_TAG_BY_ID = {
    "STRIKE": "S",
    "DEFEND": "D",
    "BASH": "B",
    "BLOODLETTING": "L",
    "INFLAME": "I",
    "TWIN_STRIKE": "T",
}
_ID_BY_TAG = {v: k for k, v in _TAG_BY_ID.items()}


def pile_to_tags(pile: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_TAG_BY_ID.get(c, c[:1]) for c in pile)


def tag_to_pile(tags: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_ID_BY_TAG.get(t, t) for t in tags)

# Extra card_ids (non-starter) sort after starter tags when needed.
_CARD_RANK: dict[str, int] = {
    "BASH": 0,
    "BLOODLETTING": 1,
    "DEFEND": 2,
    "INFLAME": 3,
    "STRIKE": 4,
    "TWIN_STRIKE": 5,
    "UPPERCUT": 6,
}


def sort_discard_pile(pile: tuple[str, ...]) -> tuple[str, ...]:
    """Canonical discard order — same multiset order as Sim 3 tag piles."""
    if not pile:
        return ()
    tags = sort_tag_pile(pile_to_tags(pile))
    return tag_to_pile(tags)


def shuffle_discard_into_draw(
    discard: tuple[str, ...],
    seed: int,
    shuffle_count: int,
) -> tuple[str, ...]:
    """Shuffle discard into draw pile using tag RNG (matches Sim 3)."""
    tags = sort_tag_pile(pile_to_tags(discard))
    shuffled = canonical_shuffle(tags, seed, shuffle_count)
    return tag_to_pile(shuffled)


def count_in_pile(pile: tuple[str, ...], card_id: str) -> int:
    return pile.count(card_id)


def remaining_in_deck(
    hand: tuple[tuple[str, int], ...],
    draw: tuple[str, ...],
    discard: tuple[str, ...],
    card_id: str,
) -> int:
    h = sum(v for k, v in hand if k == card_id)
    return h + count_in_pile(draw, card_id) + count_in_pile(discard, card_id)
