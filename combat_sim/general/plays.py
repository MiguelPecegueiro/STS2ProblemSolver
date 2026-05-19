"""Energy-bounded play enumeration and play order for the general DP."""

from __future__ import annotations

from combat_sim.card_effect import CardEffect
from combat_sim.general.hand import HandCounts, PlayCounts, play_cards_played


def play_energy_ok(
    play: PlayCounts,
    effects: dict[str, CardEffect],
    *,
    base_energy: int = 3,
) -> bool:
    """
    Legal when energy spent <= base + energy gained by cards in this play.

    energy_available = base_energy + sum(energy_gain * count)
    energy_spent     = sum(cost * count)
    """
    gain = 0
    spent = 0
    for cid, n in play.items():
        if n <= 0:
            continue
        eff = effects[cid]
        gain += eff.energy_gain * n
        spent += eff.cost * n
    return spent <= base_energy + gain


def legal_plays(
    hand: HandCounts,
    effects: dict[str, CardEffect],
    *,
    base_energy: int = 3,
) -> list[PlayCounts]:
    """
    Enumerate every sub-multiset of ``hand`` whose total energy cost is feasible.

    Each returned play maps card_id -> count (omitted keys mean 0).
    Includes the empty play (end turn with no cards).
    """
    cards = sorted(hand.keys())
    if not cards:
        return [{}]

    out: list[PlayCounts] = []

    def recurse(i: int, current: PlayCounts) -> None:
        if i == len(cards):
            if play_energy_ok(current, effects, base_energy=base_energy):
                out.append(dict(current))
            return
        cid = cards[i]
        max_n = hand[cid]
        for n in range(0, max_n + 1):
            current[cid] = n
            recurse(i + 1, current)
        current.pop(cid, None)

    recurse(0, {})
    return out


def play_order(play: PlayCounts, effects: dict[str, CardEffect]) -> list[str]:
    """
    Fixed Phase-1 ordering (matches Sim 3 tuple DP):

    energy_gain DESC → block DESC → damage DESC → card_id

    Block before damage so Slow stacks apply to attacks; higher-damage attacks
  before lower (Bash before Strike) so vuln from Bash applies to Strikes.
    """
    expanded: list[str] = []
    for cid, n in sorted(play.items()):
        if n <= 0:
            continue
        expanded.extend([cid] * n)

    def sort_key(cid: str) -> tuple:
        eff = effects[cid]
        return (-eff.energy_gain, -eff.block, -eff.damage, cid)

    expanded.sort(key=sort_key)
    return expanded


def play_has_damage(play: PlayCounts, effects: dict[str, CardEffect]) -> bool:
    for cid, n in play.items():
        if n > 0 and effects[cid].damage > 0:
            return True
    return False


def play_has_block(play: PlayCounts, effects: dict[str, CardEffect]) -> bool:
    for cid, n in play.items():
        if n > 0 and effects[cid].block > 0:
            return True
    return False


def play_hp_loss(play: PlayCounts, effects: dict[str, CardEffect]) -> int:
    total = 0
    for cid, n in play.items():
        if n > 0:
            total += effects[cid].hp_loss * n
    return total


def cards_played(play: PlayCounts) -> int:
    return play_cards_played(play)
