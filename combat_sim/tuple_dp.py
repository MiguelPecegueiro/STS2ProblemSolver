"""Pure tuple DP core — fast exact solver for Sim 0."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from combat_sim.bounds import apply_forward_prunes, hp_ceiling
from combat_sim.math_solver import TurnComposition
from combat_sim.shuffle import canonical_shuffle, sort_tag_pile

LOSS_HP = -1_000_000_000
STRIKE_DAMAGE = 6
DEFEND_BLOCK = 5
MAX_ENERGY = 3
HAND_DRAW = 5

# Intent encoding in pattern tuples: ("A", damage) or ("B", block)
IntentTuple = tuple[str, int]
Pattern = tuple[IntentTuple, ...]


@dataclass(frozen=True, slots=True)
class DpValue:
    final_hp: int
    turns_to_end: int

    @property
    def winnable(self) -> bool:
        return self.final_hp > LOSS_HP


@dataclass(frozen=True, slots=True)
class TupleState:
    hp_p: int
    hp_e: int
    block_e: int
    pattern_idx: int
    hand_s: int
    hand_d: int
    draw: tuple[str, ...]
    discard: tuple[str, ...]
    turn: int
    shuffles: int = 0

    def key(self) -> tuple:
        return (
            self.hp_p,
            self.hp_e,
            self.block_e,
            self.pattern_idx,
            self.hand_s,
            self.hand_d,
            self.draw,
            self.discard,
            self.turn,
            self.shuffles,
        )


@dataclass(frozen=True, slots=True)
class FightParams:
    player_hp: int
    enemy_hp: int
    pattern: Pattern
    max_turns: int = 30


def make_opening_state(
    params: FightParams,
    *,
    hand_s: int,
    hand_d: int,
    draw: tuple[str, ...],
) -> TupleState:
    return TupleState(
        hp_p=params.player_hp,
        hp_e=params.enemy_hp,
        block_e=0,
        pattern_idx=0,
        hand_s=hand_s,
        hand_d=hand_d,
        draw=draw,
        discard=(),
        turn=1,
    )


def state_from_engine(state, pattern: Pattern) -> TupleState:
    """Build tuple state from CombatState (start of player turn)."""
    from combat_sim.deck_counts import hand_counts

    enemy = state.living_enemies()[0] if state.living_enemies() else state.enemies[0]
    hc = hand_counts(state)

    def pile(cards) -> tuple[str, ...]:
        return tuple("S" if c.definition.damage > 0 else "D" for c in cards)

    return TupleState(
        hp_p=state.player_hp,
        hp_e=enemy.hp,
        block_e=enemy.block,
        pattern_idx=enemy.pattern_index % max(len(pattern), 1),
        hand_s=hc.strikes,
        hand_d=hc.defends,
        draw=pile(state.draw_pile),
        discard=pile(state.discard_pile),
        turn=state.turn,
        shuffles=state.shuffle_count,
    )


def feasible_plays(hand_s: int, hand_d: int) -> list[TurnComposition]:
    out: list[TurnComposition] = []
    for total in range(0, min(MAX_ENERGY, hand_s + hand_d) + 1):
        for s in range(0, min(total, hand_s, MAX_ENERGY) + 1):
            d = total - s
            if d <= hand_d:
                out.append(TurnComposition(s, d))
    return out


def ordered_plays(st: TupleState, pattern: Pattern) -> list[TurnComposition]:
    plays = feasible_plays(st.hand_s, st.hand_d)
    incoming = 0
    if pattern:
        kind, val = pattern[st.pattern_idx % len(pattern)]
        if kind == "A":
            incoming = val

    filtered: list[TurnComposition] = []
    for comp in plays:
        if comp.cards_played == 0 and st.hand_s > 0 and st.hp_e > 0:
            continue
        if comp.defends > 0 and incoming == 0 and comp.strikes == 0:
            continue
        filtered.append(comp)

    filtered.sort(key=lambda c: (-c.strikes, -c.defends, c.cards_played))
    return filtered or [TurnComposition(0, 0)]


def apply_turn(
    st: TupleState,
    comp: TurnComposition,
    pattern: Pattern,
    shuffle_seed: int,
) -> TupleState | None:
    """Play comp then enemy phase, draw, next player turn. None = player died."""
    if comp.strikes > st.hand_s or comp.defends > st.hand_d:
        return None

    hp_p = st.hp_p
    hp_e = st.hp_e
    block_e = st.block_e
    block_p = comp.defends * DEFEND_BLOCK

    dmg = comp.strikes * STRIKE_DAMAGE
    if block_e > 0:
        absorbed = min(block_e, dmg)
        block_e -= absorbed
        dmg -= absorbed
    if dmg > 0:
        hp_e = max(0, hp_e - dmg)

    idx = st.pattern_idx
    if hp_e > 0 and pattern:
        kind, val = pattern[idx % len(pattern)]
        if kind == "A":
            hp_p -= max(0, val - block_p)
        elif kind == "B":
            block_e += val
        idx = (idx + 1) % len(pattern)

    if hp_p <= 0:
        return None

    if hp_e <= 0:
        return TupleState(
            hp_p=hp_p,
            hp_e=0,
            block_e=block_e,
            pattern_idx=idx,
            hand_s=0,
            hand_d=0,
            draw=st.draw,
            discard=st.discard,
            turn=st.turn,
        )

    hand_s = st.hand_s - comp.strikes
    hand_d = st.hand_d - comp.defends
    discard = st.discard + ("S",) * comp.strikes + ("D",) * comp.defends
    discard = sort_tag_pile(discard + ("S",) * hand_s + ("D",) * hand_d)

    draw = st.draw
    hand_s, hand_d = 0, 0
    shuffles = st.shuffles
    for _ in range(HAND_DRAW):
        if not draw:
            if not discard:
                break
            draw = canonical_shuffle(discard, shuffle_seed, shuffles)
            shuffles += 1
            discard = ()
        if draw:
            card = draw[-1]
            draw = draw[:-1]
            if card == "S":
                hand_s += 1
            else:
                hand_d += 1

    return TupleState(
        hp_p=hp_p,
        hp_e=hp_e,
        block_e=block_e,
        pattern_idx=idx,
        hand_s=hand_s,
        hand_d=hand_d,
        draw=draw,
        discard=discard,
        turn=st.turn + 1,
        shuffles=shuffles,
    )


def _rank(v: DpValue) -> tuple:
    return (v.winnable, v.final_hp, -v.turns_to_end)


def solve_tuple(
    st: TupleState,
    params: FightParams,
    shuffle_seed: int,
    *,
    memo: dict[tuple, DpValue] | None = None,
    use_prunes: bool = True,
) -> DpValue:
    table: dict[tuple, DpValue] = memo if memo is not None else {}
    best_known = [LOSS_HP]
    return _solve(st, params, shuffle_seed, table, best_known, use_prunes)


def _solve(
    st: TupleState,
    params: FightParams,
    shuffle_seed: int,
    memo: dict[tuple, DpValue],
    best_known: list[int],
    use_prunes: bool,
    *,
    share_play_bound: bool = False,
) -> DpValue:
    if st.hp_p <= 0:
        return DpValue(LOSS_HP, 0)
    if st.hp_e <= 0:
        return DpValue(st.hp_p, 0)
    if st.turn > params.max_turns:
        return DpValue(LOSS_HP, 0)

    key = st.key()
    if key in memo:
        return memo[key]

    if use_prunes:
        reason = apply_forward_prunes(st, params, best_known[0])
        if reason == "survival" or reason == "kill":
            result = DpValue(LOSS_HP, 0)
            memo[key] = result
            return result
        if reason == "ceiling":
            return DpValue(LOSS_HP, 0)

    best = DpValue(LOSS_HP, 9999)
    for comp in ordered_plays(st, params.pattern):
        bound = best_known if share_play_bound else [LOSS_HP]
        nxt = apply_turn(st, comp, params.pattern, shuffle_seed)
        if nxt is None:
            child = DpValue(LOSS_HP, 1)
        elif nxt.hp_e <= 0:
            child = DpValue(nxt.hp_p, 1)
        else:
            sub = _solve(
                nxt, params, shuffle_seed, memo, bound, use_prunes, share_play_bound=True
            )
            child = DpValue(sub.final_hp, 1 + sub.turns_to_end)

        if _rank(child) > _rank(best):
            best = child
        if best.winnable and best.final_hp > best_known[0]:
            best_known[0] = best.final_hp

    memo[key] = best
    return best


def solve_with_best_play(
    st: TupleState,
    params: FightParams,
    shuffle_seed: int,
    *,
    memo: dict[tuple, tuple[DpValue, TurnComposition | None]] | None = None,
    use_prunes: bool = True,
) -> tuple[DpValue, TurnComposition | None]:
    """Return optimal value and best (strikes, defends) for this state."""
    table: dict[tuple, tuple[DpValue, TurnComposition | None]] = memo if memo is not None else {}
    best_known = [LOSS_HP]
    return _solve_with_policy(
        st, params, shuffle_seed, table, best_known, use_prunes
    )


def _solve_with_policy(
    st: TupleState,
    params: FightParams,
    shuffle_seed: int,
    memo: dict[tuple, tuple[DpValue, TurnComposition | None]],
    best_known: list[int],
    use_prunes: bool,
    *,
    share_play_bound: bool = False,
) -> tuple[DpValue, TurnComposition | None]:
    if st.hp_p <= 0:
        return DpValue(LOSS_HP, 0), None
    if st.hp_e <= 0:
        return DpValue(st.hp_p, 0), None
    if st.turn > params.max_turns:
        return DpValue(LOSS_HP, 0), None

    key = st.key()
    if key in memo:
        return memo[key]

    if use_prunes:
        reason = apply_forward_prunes(st, params, best_known[0])
        if reason == "survival" or reason == "kill":
            entry = (DpValue(LOSS_HP, 0), None)
            memo[key] = entry
            return entry
        if reason == "ceiling":
            return DpValue(LOSS_HP, 0), None

    best = DpValue(LOSS_HP, 9999)
    best_comp: TurnComposition | None = None
    for comp in ordered_plays(st, params.pattern):
        bound = best_known if share_play_bound else [LOSS_HP]
        nxt = apply_turn(st, comp, params.pattern, shuffle_seed)
        if nxt is None:
            child = DpValue(LOSS_HP, 1)
        elif nxt.hp_e <= 0:
            child = DpValue(nxt.hp_p, 1)
        else:
            sub, _ = _solve_with_policy(
                nxt,
                params,
                shuffle_seed,
                memo,
                bound,
                use_prunes,
                share_play_bound=True,
            )
            child = DpValue(sub.final_hp, 1 + sub.turns_to_end)

        if _rank(child) > _rank(best):
            best = child
            best_comp = comp
        if best.winnable and best.final_hp > best_known[0]:
            best_known[0] = best.final_hp

    entry = (best, best_comp)
    memo[key] = entry
    return entry


# --- Opening hand exact aggregation ---

OPENING_TOTAL = comb(9, 5)


def opening_hand_probability(hand_s: int, hand_d: int) -> float:
    if hand_s < 0 or hand_d < 0 or hand_s + hand_d != 5:
        return 0.0
    if hand_s > 5 or hand_d > 4:
        return 0.0
    return comb(5, hand_s) * comb(4, hand_d) / OPENING_TOTAL


def opening_compositions() -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for hand_s in range(6):
        hand_d = 5 - hand_s
        p = opening_hand_probability(hand_s, hand_d)
        if p > 0:
            rows.append((hand_s, hand_d, p))
    return rows


def remaining_deck(hand_s: int, hand_d: int) -> tuple[str, ...]:
    """Cards not in opening hand (strikes then defends, stable order)."""
    rem_s = 5 - hand_s
    rem_d = 4 - hand_d
    return ("S",) * rem_s + ("D",) * rem_d


def draw_orders(draw: tuple[str, ...]) -> list[tuple[str, ...]]:
    """All permutations of draw pile (exact over order uncertainty)."""
    if not draw:
        return [()]
    from itertools import permutations

    return list(dict.fromkeys(permutations(draw)))


@dataclass(frozen=True, slots=True)
class OpeningExpectation:
    win_rate: float
    expected_final_hp: float
    by_composition: list[tuple[int, int, float, float, float]]
    draw_orders_averaged: bool

    def format_report(self, scenario: str = "fight") -> str:
        lines = [
            "=" * 60,
            f"EXACT OPENING EXPECTATION — {scenario}",
            "=" * 60,
            f"Win rate (exact):     {self.win_rate * 100:.4f}%",
            f"E[final HP]:          {self.expected_final_hp:.4f}",
            f"Draw order averaging: {'yes' if self.draw_orders_averaged else 'canonical pile'}",
            "",
            f"{'Strikes':>7} {'Defends':>7} {'P(hand)':>10} {'P(win)':>10} {'E[HP]':>10}",
        ]
        for hs, hd, prob, wr, ehp in self.by_composition:
            lines.append(
                f"{hs:>7} {hd:>7} {prob * 100:>9.2f}% {wr * 100:>9.2f}% {ehp:>10.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def exact_opening_expectation(
    params: FightParams,
    *,
    shuffle_seed: int = 0,
    average_draw_orders: bool = True,
) -> OpeningExpectation:
    """
    E[HP] = sum_h P(h) * V*(s0(h)); win rate = sum_h P(h) * 1[V* > LOSS].
    """
    e_hp = 0.0
    win_rate = 0.0
    rows: list[tuple[int, int, float, float, float]] = []

    for hand_s, hand_d, prob in opening_compositions():
        draw_base = remaining_deck(hand_s, hand_d)
        orders = draw_orders(draw_base) if average_draw_orders else [draw_base]

        vals: list[DpValue] = []
        for draw in orders:
            st = make_opening_state(params, hand_s=hand_s, hand_d=hand_d, draw=draw)
            vals.append(
                solve_tuple(
                    st,
                    params,
                    shuffle_seed + hand_s * 10 + hand_d + hash(draw) % 1000,
                )
            )

        avg_hp = sum(v.final_hp if v.winnable else LOSS_HP for v in vals) / len(vals)
        avg_win = sum(1.0 if v.winnable else 0.0 for v in vals) / len(vals)

        e_hp += prob * avg_hp
        win_rate += prob * avg_win
        rows.append((hand_s, hand_d, prob, avg_win, avg_hp))

    return OpeningExpectation(
        win_rate=win_rate,
        expected_final_hp=e_hp,
        by_composition=rows,
        draw_orders_averaged=average_draw_orders,
    )
