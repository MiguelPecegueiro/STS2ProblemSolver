"""Admissible forward-pruning bounds for Sim 0 tuple DP."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from combat_sim.tuple_dp import FightParams, Pattern, TupleState

LOSS_HP = -1_000_000_000
STRIKE_DAMAGE = 6
DEFEND_BLOCK = 5
MAX_ENERGY = 3


def count_strikes_defends(
    hand_s: int,
    hand_d: int,
    draw: tuple[str, ...],
    discard: tuple[str, ...],
) -> tuple[int, int]:
    s = hand_s + draw.count("S") + discard.count("S")
    d = hand_d + draw.count("D") + discard.count("D")
    return s, d


def _attack_damage_at(pattern: Pattern, pattern_idx: int, turn_offset: int) -> int:
    if not pattern:
        return 0
    kind, val = pattern[(pattern_idx + turn_offset) % len(pattern)]
    if kind == "A":
        return val
    return 0


def block_max_per_turn(hand_s: int, hand_d: int, draw: tuple[str, ...]) -> int:
    """Best-case block per turn from hand + draw defends (user formula)."""
    return DEFEND_BLOCK * min(MAX_ENERGY, hand_d + draw.count("D"))


def damage_min_survival(
    st: TupleState,  # type: ignore[name-defined]
    pattern: Pattern,  # type: ignore[name-defined]
    max_turns: int,
) -> int:
    """
    Minimum damage taken from this turn onward if we block optimally each turn
    (optimistic block cap per turn from current hand+draw defends).
    """
    total = 0
    turns_left = max(0, max_turns - st.turn + 1)
    for k in range(turns_left):
        atk = _attack_damage_at(pattern, st.pattern_idx, k)
        if atk > 0:
            blk = block_max_per_turn(st.hand_s, st.hand_d, st.draw)
            total += max(0, atk - blk)
    return total


def damage_unavoidable(
    st: TupleState,
    pattern: Pattern,
    max_turns: int,
) -> int:
    """Same as damage_min_survival for Sim 0 (minimum damage we must absorb)."""
    return damage_min_survival(st, pattern, max_turns)


def damage_max_kill(
    st: TupleState,
    max_turns: int,
    *,
    hp_loss_cap_per_turn: int | None = None,
) -> int:
    """Maximum strike damage dealable over remaining player turns."""
    from combat_sim.damage_util import max_kill_total

    s, _ = count_strikes_defends(st.hand_s, st.hand_d, st.draw, st.discard)
    turns_left = max(0, max_turns - st.turn + 1)
    per_turn = STRIKE_DAMAGE * min(MAX_ENERGY, s)
    return max_kill_total(per_turn, turns_left, hp_loss_cap_per_turn)


def hp_ceiling(st: TupleState, pattern: Pattern, max_turns: int) -> int:
    """Upper bound on final player HP achievable from this state."""
    return st.hp_p - damage_unavoidable(st, pattern, max_turns)


def prune_survival(st: TupleState, pattern: Pattern, max_turns: int) -> bool:
    """Bound 1: cannot survive even with perfect block."""
    if st.hp_e <= 0:
        return False
    dmg = damage_min_survival(st, pattern, max_turns)
    return st.hp_p - dmg <= 0


def prune_unwinnable(st: TupleState, max_turns: int) -> bool:
    """Bound 2: cannot kill enemy even with perfect damage."""
    if st.hp_e <= 0:
        return False
    dmg = damage_max_kill(st, max_turns)
    return st.hp_e > dmg


def prune_suboptimal(
    st: TupleState,
    pattern: Pattern,
    max_turns: int,
    best_known_hp: int,
) -> bool:
    """Bound 3: best-case HP from here cannot beat known optimum."""
    if best_known_hp <= LOSS_HP:
        return False
    return hp_ceiling(st, pattern, max_turns) < best_known_hp


def apply_forward_prunes(
    st: "TupleState",
    params: "FightParams",
    best_known_hp: int,
) -> str | None:
    """
    Return prune reason if state can be cut, else None.
    All conditions are admissible for max-final-HP objective.
    """
    if prune_survival(st, params.pattern, params.max_turns):
        return "survival"
    if prune_unwinnable(st, params.max_turns):
        return "kill"
    if prune_suboptimal(st, params.pattern, params.max_turns, best_known_hp):
        return "ceiling"
    return None
