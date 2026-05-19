"""Hand and play multisets keyed by card_id."""

from __future__ import annotations

HandCounts = dict[str, int]
PlayCounts = dict[str, int]


def counts_from_tuple(items: tuple[tuple[str, int], ...]) -> HandCounts:
    return {k: v for k, v in items if v > 0}


def counts_to_tuple(counts: HandCounts) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((k, v) for k, v in counts.items() if v > 0))


def subtract_hand(hand: HandCounts, play: PlayCounts) -> HandCounts:
    out = dict(hand)
    for cid, n in play.items():
        if n <= 0:
            continue
        out[cid] = out.get(cid, 0) - n
        if out[cid] <= 0:
            out.pop(cid, None)
    return out


def total_cards(counts: HandCounts) -> int:
    return sum(counts.values())


def play_cards_played(play: PlayCounts) -> int:
    return sum(play.values())
