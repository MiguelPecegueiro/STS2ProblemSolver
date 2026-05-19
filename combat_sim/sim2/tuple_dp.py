"""Pure tuple DP for Sim 2 (Bloodletting + dynamic energy)."""

from __future__ import annotations

from dataclasses import dataclass

from combat_sim.shuffle import canonical_shuffle, sort_tag_pile
from combat_sim.damage_util import calc_damage
from combat_sim.sim1.damage import (
    BASH_DAMAGE,
    STRIKE_DAMAGE,
    apply_damage_to_enemy,
    damage_with_vulnerable,
)
from combat_sim.sim2.composition import (
    BASE_ENERGY,
    ENERGY_PER_BLOODLETTING,
    TurnCompositionSim2,
    feasible_plays_sim2,
)
from combat_sim.pattern_util import parse_pattern_step
from combat_sim.tuple_dp import DEFEND_BLOCK, HAND_DRAW, LOSS_HP, Pattern

VULN_APPLY = 2
BLOODLETTING_HP_LOSS = 3
MAX_ENERGY_WITH_BL = BASE_ENERGY + ENERGY_PER_BLOODLETTING


@dataclass(frozen=True, slots=True)
class DpValue:
    final_hp: int
    turns_to_end: int

    @property
    def winnable(self) -> bool:
        return self.final_hp > LOSS_HP


@dataclass(frozen=True, slots=True)
class TupleStateSim2:
    hp_p: int
    hp_e: int
    block_e: int
    vuln_e: int
    pattern_idx: int
    hand_s: int
    hand_d: int
    hand_b: int
    hand_bl: int
    draw: tuple[str, ...]
    discard: tuple[str, ...]
    turn: int
    shuffles: int = 0

    def key(self) -> tuple:
        return (
            self.hp_p,
            self.hp_e,
            self.block_e,
            self.vuln_e,
            self.pattern_idx,
            self.hand_s,
            self.hand_d,
            self.hand_b,
            self.hand_bl,
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
    enemy_slow: bool = False


def make_opening_state(
    params: FightParams,
    *,
    hand_s: int,
    hand_d: int,
    hand_b: int,
    hand_bl: int,
    draw: tuple[str, ...],
) -> TupleStateSim2:
    return TupleStateSim2(
        hp_p=params.player_hp,
        hp_e=params.enemy_hp,
        block_e=0,
        vuln_e=0,
        pattern_idx=0,
        hand_s=hand_s,
        hand_d=hand_d,
        hand_b=hand_b,
        hand_bl=hand_bl,
        draw=draw,
        discard=(),
        turn=1,
    )


def state_from_engine(state, pattern: Pattern) -> TupleStateSim2:
    from combat_sim.deck_counts import hand_counts_sim2

    enemy = state.living_enemies()[0] if state.living_enemies() else state.enemies[0]
    hc = hand_counts_sim2(state)

    def pile(cards) -> tuple[str, ...]:
        from combat_sim.shuffle import card_tag

        return tuple(card_tag(c.definition) for c in cards)

    return TupleStateSim2(
        hp_p=state.player_hp,
        hp_e=enemy.hp,
        block_e=enemy.block,
        vuln_e=enemy.vuln_stacks,
        pattern_idx=enemy.pattern_index % max(len(pattern), 1),
        hand_s=hc.strikes,
        hand_d=hc.defends,
        hand_b=hc.bash,
        hand_bl=hc.bloodletting,
        draw=pile(state.draw_pile),
        discard=pile(state.discard_pile),
        turn=state.turn,
        shuffles=state.shuffle_count,
    )


def _apply_player_plays(
    hp_p: int,
    hp_e: int,
    block_e: int,
    vuln_e: int,
    comp: TurnCompositionSim2,
    *,
    enemy_slow: bool = False,
) -> tuple[int, int, int, int, int]:
    """Order: Bloodletting (hp/energy) -> Defend block -> Bash -> Strikes."""
    slow_n = 0

    for _ in range(comp.bloodletting):
        hp_p -= BLOODLETTING_HP_LOSS
        if enemy_slow:
            slow_n += 1

    block_p = comp.defends * DEFEND_BLOCK
    if enemy_slow:
        slow_n += comp.defends

    for _ in range(comp.bash):
        slow = slow_n if enemy_slow else 0
        dmg = calc_damage(BASH_DAMAGE, 0, vuln_e, slow)
        hp_e, block_e = apply_damage_to_enemy(hp_e, block_e, dmg)
        vuln_e += VULN_APPLY
        if enemy_slow:
            slow_n += 1

    for _ in range(comp.strikes):
        slow = slow_n if enemy_slow else 0
        dmg = calc_damage(STRIKE_DAMAGE, 0, vuln_e, slow)
        hp_e, block_e = apply_damage_to_enemy(hp_e, block_e, dmg)
        if enemy_slow:
            slow_n += 1

    return hp_p, hp_e, block_e, vuln_e, block_p


def apply_turn_sim2(
    st: TupleStateSim2,
    comp: TurnCompositionSim2,
    pattern: Pattern,
    shuffle_seed: int,
    *,
    enemy_slow: bool = False,
) -> TupleStateSim2 | None:
    if (
        comp.strikes > st.hand_s
        or comp.defends > st.hand_d
        or comp.bash > st.hand_b
        or comp.bloodletting > st.hand_bl
    ):
        return None
    if comp.energy_cost > comp.energy_available:
        return None

    hp_p, hp_e, block_e, vuln_e, block_p = _apply_player_plays(
        st.hp_p, st.hp_e, st.block_e, st.vuln_e, comp, enemy_slow=enemy_slow
    )

    if hp_p <= 0:
        return None

    idx = st.pattern_idx
    if hp_e > 0 and pattern:
        kind, val, bonus_block = parse_pattern_step(pattern[idx % len(pattern)])
        if kind == "A":
            hp_p -= max(0, val - block_p)
            block_e += bonus_block
        elif kind == "B":
            block_e += val
        idx = (idx + 1) % len(pattern)

    if hp_p <= 0:
        return None

    if hp_e <= 0:
        return TupleStateSim2(
            hp_p=hp_p,
            hp_e=0,
            block_e=block_e,
            vuln_e=vuln_e,
            pattern_idx=idx,
            hand_s=0,
            hand_d=0,
            hand_b=0,
            hand_bl=0,
            draw=st.draw,
            discard=st.discard,
            turn=st.turn,
            shuffles=st.shuffles,
        )

    hand_s = st.hand_s - comp.strikes
    hand_d = st.hand_d - comp.defends
    hand_b = st.hand_b - comp.bash
    hand_bl = st.hand_bl - comp.bloodletting
    discard = (
        st.discard
        + ("L",) * comp.bloodletting
        + ("D",) * comp.defends
        + ("B",) * comp.bash
        + ("S",) * comp.strikes
        + ("L",) * hand_bl
        + ("D",) * hand_d
        + ("B",) * hand_b
        + ("S",) * hand_s
    )
    discard = sort_tag_pile(discard)

    draw = st.draw
    hand_s, hand_d, hand_b, hand_bl = 0, 0, 0, 0
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
            elif card == "B":
                hand_b += 1
            elif card == "L":
                hand_bl += 1
            else:
                hand_d += 1

    vuln_e = max(0, vuln_e - 1)

    return TupleStateSim2(
        hp_p=hp_p,
        hp_e=hp_e,
        block_e=block_e,
        vuln_e=vuln_e,
        pattern_idx=idx,
        hand_s=hand_s,
        hand_d=hand_d,
        hand_b=hand_b,
        hand_bl=hand_bl,
        draw=draw,
        discard=discard,
        turn=st.turn + 1,
        shuffles=shuffles,
    )


def ordered_plays_sim2(st: TupleStateSim2, pattern: Pattern) -> list[TurnCompositionSim2]:
    plays = feasible_plays_sim2(st.hand_s, st.hand_d, st.hand_b, st.hand_bl)
    incoming = 0
    if pattern:
        kind, val = pattern[st.pattern_idx % len(pattern)]
        if kind == "A":
            incoming = val

    filtered: list[TurnCompositionSim2] = []
    for comp in plays:
        if comp.bloodletting > 0 and st.hp_p - BLOODLETTING_HP_LOSS * comp.bloodletting <= 0:
            continue
        if (
            comp.cards_played == 0
            and (st.hand_s > 0 or st.hand_b > 0 or st.hand_bl > 0)
            and st.hp_e > 0
        ):
            continue
        if comp.defends > 0 and incoming == 0 and comp.strikes == 0 and comp.bash == 0:
            if comp.bloodletting == 0:
                continue
        if (
            comp.cards_played == 0
            and st.hp_e > 0
            and (
                st.hand_s > 0
                or st.hand_b > 0
                or st.hand_bl > 0
                or (st.hand_d > 0 and incoming > 0)
            )
        ):
            continue
        filtered.append(comp)

    filtered.sort(
        key=lambda c: (-c.defends, -c.bloodletting, -c.bash, -c.strikes, -c.cards_played)
    )
    return filtered or [TurnCompositionSim2(0, 0, 0, 0)]


def _rank(v: DpValue) -> tuple:
    return (v.winnable, v.final_hp, -v.turns_to_end)


def _attack_damage_at(pattern: Pattern, pattern_idx: int, turn_offset: int) -> int:
    if not pattern:
        return 0
    kind, val = pattern[(pattern_idx + turn_offset) % len(pattern)]
    if kind == "A":
        return val
    return 0


def _bloodletting_remaining(st: TupleStateSim2) -> int:
    return st.hand_bl + st.draw.count("L") + st.discard.count("L")


def _max_energy_available(st: TupleStateSim2) -> int:
    if _bloodletting_remaining(st) > 0:
        return MAX_ENERGY_WITH_BL
    return BASE_ENERGY


def _block_max_per_turn(st: TupleStateSim2) -> int:
    return DEFEND_BLOCK * min(_max_energy_available(st), st.hand_d + st.draw.count("D"))


def _damage_min_survival(st: TupleStateSim2, params: FightParams) -> int:
    bl_rem = _bloodletting_remaining(st)
    self_dmg = BLOODLETTING_HP_LOSS * bl_rem
    total = self_dmg
    turns_left = max(0, params.max_turns - st.turn + 1)
    for k in range(turns_left):
        atk = _attack_damage_at(params.pattern, st.pattern_idx, k)
        if atk > 0:
            total += max(0, atk - _block_max_per_turn(st))
    return total


def _damage_max_kill(st: TupleStateSim2, max_turns: int) -> int:
    s = st.hand_s + st.draw.count("S") + st.discard.count("S")
    b = st.hand_b + st.draw.count("B") + st.discard.count("B")
    energy = _max_energy_available(st)
    turns_left = max(0, max_turns - st.turn + 1)
    strike_cap = min(energy, s)
    per_turn = int(STRIKE_DAMAGE * strike_cap * 1.5)
    if b > 0:
        per_turn += int(damage_with_vulnerable(BASH_DAMAGE, 1) * min(1, b))
    return per_turn * turns_left


def _hp_ceiling(st: TupleStateSim2, params: FightParams) -> int:
    return st.hp_p - _damage_min_survival(st, params)


def _apply_forward_prunes(st: TupleStateSim2, params: FightParams) -> str | None:
    if st.hp_e <= 0:
        return None
    max_kill = _damage_max_kill(st, params.max_turns)
    if st.hp_p - _damage_min_survival(st, params) <= 0 and st.hp_e > max_kill:
        return "survival"
    if st.hp_e > max_kill:
        return "kill"
    return None


def _ceiling_prune(st: TupleStateSim2, params: FightParams, best_known_hp: int) -> bool:
    if best_known_hp <= LOSS_HP:
        return False
    return _hp_ceiling(st, params) < best_known_hp


MemoEntry = tuple[DpValue, TurnCompositionSim2 | None]


def _child_value(
    st: TupleStateSim2,
    comp: TurnCompositionSim2,
    params: FightParams,
    shuffle_seed: int,
    memo: dict[tuple, MemoEntry],
    best_known: list[int],
    use_prunes: bool,
) -> DpValue:
    nxt = apply_turn_sim2(
        st, comp, params.pattern, shuffle_seed, enemy_slow=params.enemy_slow
    )
    if nxt is None:
        return DpValue(LOSS_HP, 1)
    if nxt.hp_e <= 0:
        return DpValue(nxt.hp_p, 1)
    sub, _ = _solve_core(
        nxt, params, shuffle_seed, memo, best_known, use_prunes, share_play_bound=True
    )
    return DpValue(sub.final_hp, 1 + sub.turns_to_end)


def _solve_core(
    st: TupleStateSim2,
    params: FightParams,
    shuffle_seed: int,
    memo: dict[tuple, MemoEntry],
    best_known: list[int],
    use_prunes: bool,
    *,
    share_play_bound: bool = False,
) -> MemoEntry:
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
        reason = _apply_forward_prunes(st, params)
        if reason in ("survival", "kill"):
            entry: MemoEntry = (DpValue(LOSS_HP, 0), TurnCompositionSim2(0, 0, 0, 0))
            memo[key] = entry
            return entry
        if _ceiling_prune(st, params, best_known[0]):
            return DpValue(LOSS_HP, 0), None

    best = DpValue(LOSS_HP, 9999)
    best_comp: TurnCompositionSim2 | None = None
    for comp in ordered_plays_sim2(st, params.pattern):
        bound = best_known if share_play_bound else [LOSS_HP]
        child = _child_value(st, comp, params, shuffle_seed, memo, bound, use_prunes)
        if _rank(child) > _rank(best):
            best = child
            best_comp = comp
        if best.winnable and best.final_hp > best_known[0]:
            best_known[0] = best.final_hp

    if best_comp is None and best.winnable:
        best_comp = TurnCompositionSim2(0, 0, 0, 0)

    entry = (best, best_comp)
    memo[key] = entry
    return entry


def solve_tuple_sim2(
    st: TupleStateSim2,
    params: FightParams,
    shuffle_seed: int,
    *,
    memo: dict[tuple, MemoEntry] | None = None,
    use_prunes: bool = True,
    best_known: list[int] | None = None,
) -> DpValue:
    table: dict[tuple, MemoEntry] = memo if memo is not None else {}
    bound = best_known if best_known is not None else [LOSS_HP]
    return _solve_core(st, params, shuffle_seed, table, bound, use_prunes)[0]


def solve_with_best_play_sim2(
    st: TupleStateSim2,
    params: FightParams,
    shuffle_seed: int,
    *,
    memo: dict[tuple, MemoEntry] | None = None,
    use_prunes: bool = True,
    best_known: list[int] | None = None,
) -> MemoEntry:
    table: dict[tuple, MemoEntry] = memo if memo is not None else {}
    bound = best_known if best_known is not None else [LOSS_HP]
    return _solve_core(st, params, shuffle_seed, table, bound, use_prunes)
