"""Pure tuple DP for Sim 3 (Inflame + permanent Strength)."""

from __future__ import annotations

from dataclasses import dataclass

from combat_sim.damage_util import calc_damage, optimistic_turn_attack_damage
from combat_sim.shuffle import canonical_shuffle, sort_tag_pile
from combat_sim.sim1.damage import BASH_DAMAGE, STRIKE_DAMAGE
from combat_sim.sim3.composition import (
    BASE_ENERGY,
    ENERGY_PER_BLOODLETTING,
    TurnCompositionSim3,
    feasible_plays_sim3,
)
from combat_sim.damage_util import apply_damage_to_enemy_hp_block, max_kill_total
from combat_sim.pattern_util import parse_pattern_step
from combat_sim.tuple_dp import DEFEND_BLOCK, HAND_DRAW, LOSS_HP, Pattern

VULN_APPLY = 2
BLOODLETTING_HP_LOSS = 3
INFLAME_STRENGTH = 2
MAX_ENERGY_WITH_BL = BASE_ENERGY + ENERGY_PER_BLOODLETTING


def _strength_upper_bound(st: TupleStateSim3) -> int:
    s = st.strength_p
    if st.hand_inf + st.draw.count("I") + st.discard.count("I") > 0:
        s += INFLAME_STRENGTH
    return s


def max_single_hit_damage(st: TupleStateSim3) -> int:
    """Max damage from one strike or bash (optimistic, for block memo canonicalization)."""
    str_p = _strength_upper_bound(st)
    vuln = st.vuln_e
    dmg = calc_damage(STRIKE_DAMAGE, str_p, vuln, 0)
    if st.hand_b + st.draw.count("B") + st.discard.count("B") > 0:
        dmg = max(dmg, calc_damage(BASH_DAMAGE, str_p, vuln, 0))
    return dmg


def canonical_block_e_for_key(st: TupleStateSim3) -> int:
    """Block above max next hit is equivalent for all future strike lines."""
    if st.block_e <= 0:
        return 0
    cap = max_single_hit_damage(st)
    if cap <= 0:
        return st.block_e
    return min(st.block_e, cap)


@dataclass(frozen=True, slots=True)
class DpValue:
    final_hp: int
    turns_to_end: int

    @property
    def winnable(self) -> bool:
        return self.final_hp > LOSS_HP


@dataclass(frozen=True, slots=True)
class TupleStateSim3:
    hp_p: int
    block_p: int = 0
    strength_p: int = 0
    hp_e: int = 0
    block_e: int = 0
    vuln_e: int = 0
    pattern_idx: int = 0
    hand_s: int = 0
    hand_d: int = 0
    hand_b: int = 0
    hand_bl: int = 0
    hand_inf: int = 0
    draw: tuple[str, ...] = ()
    discard: tuple[str, ...] = ()
    turn: int = 1
    shuffles: int = 0

    def key(self) -> tuple:
        return (
            self.hp_p,
            self.block_p,
            self.strength_p,
            self.hp_e,
            canonical_block_e_for_key(self),
            self.vuln_e,
            self.pattern_idx,
            self.hand_s,
            self.hand_d,
            self.hand_b,
            self.hand_bl,
            self.hand_inf,
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
    enemy_hp_loss_cap: int | None = None


def make_opening_state(
    params: FightParams,
    *,
    hand_s: int,
    hand_d: int,
    hand_b: int,
    hand_bl: int,
    hand_inf: int,
    draw: tuple[str, ...],
) -> TupleStateSim3:
    return TupleStateSim3(
        hp_p=params.player_hp,
        hp_e=params.enemy_hp,
        block_e=0,
        vuln_e=0,
        pattern_idx=0,
        hand_s=hand_s,
        hand_d=hand_d,
        hand_b=hand_b,
        hand_bl=hand_bl,
        hand_inf=hand_inf,
        draw=draw,
        discard=(),
        turn=1,
    )


def state_from_engine(state, pattern: Pattern) -> TupleStateSim3:
    from combat_sim.deck_counts import hand_counts_sim3

    enemy = state.living_enemies()[0] if state.living_enemies() else state.enemies[0]
    hc = hand_counts_sim3(state)

    def pile(cards) -> tuple[str, ...]:
        from combat_sim.shuffle import card_tag

        return tuple(card_tag(c.definition) for c in cards)

    return TupleStateSim3(
        hp_p=state.player_hp,
        block_p=state.player_block,
        strength_p=state.player_strength,
        hp_e=enemy.hp,
        block_e=enemy.block,
        vuln_e=enemy.vuln_stacks,
        pattern_idx=enemy.pattern_index % max(len(pattern), 1),
        hand_s=hc.strikes,
        hand_d=hc.defends,
        hand_b=hc.bash,
        hand_bl=hc.bloodletting,
        hand_inf=hc.inflame,
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
    strength_p: int,
    comp: TurnCompositionSim3,
    *,
    enemy_slow: bool = False,
    enemy_hp_loss_cap: int | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Order: BL -> Defend -> Inflame -> Bash -> Strike."""
    slow_n = 0
    hp_lost = 0

    def _hit(damage: int) -> None:
        nonlocal hp_e, block_e, hp_lost
        hp_e, block_e, dealt = apply_damage_to_enemy_hp_block(
            hp_e,
            block_e,
            damage,
            hp_loss_cap=enemy_hp_loss_cap,
            hp_lost_this_turn=hp_lost,
        )
        hp_lost += dealt

    for _ in range(comp.bloodletting):
        hp_p -= BLOODLETTING_HP_LOSS
        if enemy_slow:
            slow_n += 1

    block_p = comp.defends * DEFEND_BLOCK
    if enemy_slow:
        slow_n += comp.defends

    strength_p += INFLAME_STRENGTH * comp.inflame
    if enemy_slow:
        slow_n += comp.inflame

    for _ in range(comp.bash):
        slow = slow_n if enemy_slow else 0
        dmg = calc_damage(BASH_DAMAGE, strength_p, vuln_e, slow)
        _hit(dmg)
        vuln_e += VULN_APPLY
        if enemy_slow:
            slow_n += 1

    for _ in range(comp.strikes):
        slow = slow_n if enemy_slow else 0
        dmg = calc_damage(STRIKE_DAMAGE, strength_p, vuln_e, slow)
        _hit(dmg)
        if enemy_slow:
            slow_n += 1

    return hp_p, hp_e, block_e, vuln_e, strength_p, block_p


def apply_turn_sim3(
    st: TupleStateSim3,
    comp: TurnCompositionSim3,
    pattern: Pattern,
    shuffle_seed: int,
    *,
    enemy_slow: bool = False,
    enemy_hp_loss_cap: int | None = None,
) -> TupleStateSim3 | None:
    if (
        comp.strikes > st.hand_s
        or comp.defends > st.hand_d
        or comp.bash > st.hand_b
        or comp.bloodletting > st.hand_bl
        or comp.inflame > st.hand_inf
    ):
        return None
    if comp.energy_cost > comp.energy_available:
        return None

    hp_p, hp_e, block_e, vuln_e, strength_p, block_p = _apply_player_plays(
        st.hp_p,
        st.hp_e,
        st.block_e,
        st.vuln_e,
        st.strength_p,
        comp,
        enemy_slow=enemy_slow,
        enemy_hp_loss_cap=enemy_hp_loss_cap,
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
        return TupleStateSim3(
            hp_p=hp_p,
            strength_p=strength_p,
            hp_e=0,
            block_e=block_e,
            vuln_e=vuln_e,
            pattern_idx=idx,
            hand_s=0,
            hand_d=0,
            hand_b=0,
            hand_bl=0,
            hand_inf=0,
            draw=st.draw,
            discard=st.discard,
            turn=st.turn,
            shuffles=st.shuffles,
        )

    hand_s = st.hand_s - comp.strikes
    hand_d = st.hand_d - comp.defends
    hand_b = st.hand_b - comp.bash
    hand_bl = st.hand_bl - comp.bloodletting
    hand_inf = st.hand_inf - comp.inflame
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
        + ("I",) * hand_inf
    )
    discard = sort_tag_pile(discard)

    draw = st.draw
    hand_s, hand_d, hand_b, hand_bl, hand_inf = 0, 0, 0, 0, 0
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
            elif card == "I":
                hand_inf += 1
            else:
                hand_d += 1

    vuln_e = max(0, vuln_e - 1)

    return TupleStateSim3(
        hp_p=hp_p,
        strength_p=strength_p,
        hp_e=hp_e,
        block_e=block_e,
        vuln_e=vuln_e,
        pattern_idx=idx,
        hand_s=hand_s,
        hand_d=hand_d,
        hand_b=hand_b,
        hand_bl=hand_bl,
        hand_inf=hand_inf,
        draw=draw,
        discard=discard,
        turn=st.turn + 1,
        shuffles=shuffles,
    )


def ordered_plays_sim3(st: TupleStateSim3, pattern: Pattern) -> list[TurnCompositionSim3]:
    plays = feasible_plays_sim3(
        st.hand_s, st.hand_d, st.hand_b, st.hand_bl, st.hand_inf
    )
    incoming = 0
    if pattern:
        kind, val, _ = parse_pattern_step(pattern[st.pattern_idx % len(pattern)])
        if kind == "A":
            incoming = val

    filtered: list[TurnCompositionSim3] = []
    for comp in plays:
        if comp.bloodletting > 0 and st.hp_p - BLOODLETTING_HP_LOSS * comp.bloodletting <= 0:
            continue
        if (
            comp.cards_played == 0
            and (
                st.hand_s > 0
                or st.hand_b > 0
                or st.hand_bl > 0
                or st.hand_inf > 0
            )
            and st.hp_e > 0
        ):
            continue
        if comp.defends > 0 and incoming == 0 and comp.strikes == 0 and comp.bash == 0:
            if comp.bloodletting == 0 and comp.inflame == 0:
                continue
        if (
            comp.cards_played == 0
            and st.hp_e > 0
            and (
                st.hand_s > 0
                or st.hand_b > 0
                or st.hand_bl > 0
                or st.hand_inf > 0
                or (st.hand_d > 0 and incoming > 0)
            )
        ):
            continue
        filtered.append(comp)

    filtered.sort(
        key=lambda c: (
            -c.defends,
            -c.bloodletting,
            -c.inflame,
            -c.bash,
            -c.strikes,
            -c.cards_played,
        )
    )
    return filtered or [TurnCompositionSim3(0, 0, 0, 0, 0)]


def _rank(v: DpValue) -> tuple:
    return (v.winnable, v.final_hp, -v.turns_to_end)


def _attack_damage_at(pattern: Pattern, pattern_idx: int, turn_offset: int) -> int:
    if not pattern:
        return 0
    kind, val, _ = parse_pattern_step(pattern[(pattern_idx + turn_offset) % len(pattern)])
    if kind == "A":
        return val
    return 0


def _bloodletting_remaining(st: TupleStateSim3) -> int:
    return st.hand_bl + st.draw.count("L") + st.discard.count("L")


def _inflame_remaining(st: TupleStateSim3) -> int:
    return st.hand_inf + st.draw.count("I") + st.discard.count("I")


def _max_strength_for_prune(st: TupleStateSim3) -> int:
    s = st.strength_p
    if _inflame_remaining(st) > 0:
        s += INFLAME_STRENGTH
    return s


def _max_energy_available(st: TupleStateSim3) -> int:
    if _bloodletting_remaining(st) > 0:
        return MAX_ENERGY_WITH_BL
    return BASE_ENERGY


def _block_max_per_turn(st: TupleStateSim3) -> int:
    return DEFEND_BLOCK * min(_max_energy_available(st), st.hand_d + st.draw.count("D"))


def _damage_min_survival(st: TupleStateSim3, params: FightParams) -> int:
    bl_rem = _bloodletting_remaining(st)
    self_dmg = BLOODLETTING_HP_LOSS * bl_rem
    total = self_dmg
    turns_left = max(0, params.max_turns - st.turn + 1)
    for k in range(turns_left):
        atk = _attack_damage_at(params.pattern, st.pattern_idx, k)
        if atk > 0:
            total += max(0, atk - _block_max_per_turn(st))
    return total


def _damage_max_kill(
    st: TupleStateSim3,
    max_turns: int,
    *,
    enemy_slow: bool = False,
    enemy_hp_loss_cap: int | None = None,
) -> int:
    str_p = _max_strength_for_prune(st)
    s = st.hand_s + st.draw.count("S") + st.discard.count("S")
    b_avail = st.hand_b + st.draw.count("B") + st.discard.count("B")
    energy = _max_energy_available(st)
    turns_left = max(0, max_turns - st.turn + 1)
    if enemy_slow:
        bl = min(1, _bloodletting_remaining(st))
        inf = min(1, _inflame_remaining(st))
        d = st.hand_d + st.draw.count("D")
        bash = min(1, b_avail) if energy >= 2 else 0
        e_left = energy - 2 * bash
        defends = min(d, max(0, e_left))
        e_left -= defends
        inflame = min(inf, max(0, e_left))
        e_left -= inflame
        strikes = min(s, max(0, e_left))
        per_turn = optimistic_turn_attack_damage(
            strikes=strikes,
            defends=defends,
            bash=bash,
            bloodletting=bl,
            inflame=inflame,
            strength=str_p,
            vuln_stacks=1,
            bash_damage=BASH_DAMAGE,
            strike_damage=STRIKE_DAMAGE,
        )
    else:
        strike_cap = min(energy, s)
        per_turn = 0
        for _ in range(strike_cap):
            per_turn += calc_damage(STRIKE_DAMAGE, str_p, 1, 0)
        if b_avail > 0:
            per_turn += calc_damage(BASH_DAMAGE, str_p, 1, 0)
    return max_kill_total(per_turn, turns_left, enemy_hp_loss_cap)


def _hp_ceiling(st: TupleStateSim3, params: FightParams) -> int:
    return st.hp_p - _damage_min_survival(st, params)


def _apply_forward_prunes(st: TupleStateSim3, params: FightParams) -> str | None:
    from combat_sim.damage_util import prune_kill_shell_turn_budget

    if st.hp_e <= 0:
        return None
    if prune_kill_shell_turn_budget(
        st.hp_e, st.turn, params.max_turns, params.enemy_hp_loss_cap
    ):
        return "kill"
    max_kill = _damage_max_kill(
        st,
        params.max_turns,
        enemy_slow=params.enemy_slow,
        enemy_hp_loss_cap=params.enemy_hp_loss_cap,
    )
    # Survival only if pessimistic damage kills us AND we cannot kill the enemy in time.
    if st.hp_p - _damage_min_survival(st, params) <= 0 and st.hp_e > max_kill:
        return "survival"
    if st.hp_e > max_kill:
        return "kill"
    return None


def _ceiling_prune(st: TupleStateSim3, params: FightParams, best_known_hp: int) -> bool:
    if best_known_hp <= LOSS_HP:
        return False
    return _hp_ceiling(st, params) < best_known_hp


MemoEntry = tuple[DpValue, TurnCompositionSim3 | None]


def _child_value(
    st: TupleStateSim3,
    comp: TurnCompositionSim3,
    params: FightParams,
    shuffle_seed: int,
    memo: dict[tuple, MemoEntry],
    best_known: list[int],
    use_prunes: bool,
) -> DpValue:
    nxt = apply_turn_sim3(
        st,
        comp,
        params.pattern,
        shuffle_seed,
        enemy_slow=params.enemy_slow,
        enemy_hp_loss_cap=params.enemy_hp_loss_cap,
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
    st: TupleStateSim3,
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
            entry: MemoEntry = (DpValue(LOSS_HP, 0), TurnCompositionSim3(0, 0, 0, 0, 0))
            memo[key] = entry
            return entry
        if _ceiling_prune(st, params, best_known[0]):
            return DpValue(LOSS_HP, 0), None

    best = DpValue(LOSS_HP, 9999)
    best_comp: TurnCompositionSim3 | None = None
    for comp in ordered_plays_sim3(st, params.pattern):
        bound = best_known if share_play_bound else [LOSS_HP]
        child = _child_value(st, comp, params, shuffle_seed, memo, bound, use_prunes)
        if _rank(child) > _rank(best):
            best = child
            best_comp = comp
        if best.winnable and best.final_hp > best_known[0]:
            best_known[0] = best.final_hp

    if best_comp is None and best.winnable:
        best_comp = TurnCompositionSim3(0, 0, 0, 0, 0)

    entry = (best, best_comp)
    memo[key] = entry
    return entry


def solve_tuple_sim3(
    st: TupleStateSim3,
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


def solve_with_best_play_sim3(
    st: TupleStateSim3,
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
