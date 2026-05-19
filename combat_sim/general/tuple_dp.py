"""General card-id tuple DP solver."""

from __future__ import annotations

from combat_sim.general.apply import apply_turn
from combat_sim.general.hand import PlayCounts
from combat_sim.general.plays import (
    cards_played,
    legal_plays,
    play_has_block,
    play_has_damage,
    play_hp_loss,
)
from combat_sim.damage_util import calc_damage, prune_kill_shell_turn_budget
from combat_sim.general.sim3_bridge import to_sim3_state
from combat_sim.general.state import DeckContext, GeneralState, card_remaining
from combat_sim.pattern_util import parse_pattern_step
from combat_sim.sim3 import tuple_dp as sim3_dp
from combat_sim.sim3.tuple_dp import DpValue, FightParams, Pattern
from combat_sim.tuple_dp import LOSS_HP

__all__ = ["DpValue", "FightParams", "solve_tuple_general", "solve_with_best_play_general"]

_STARTER_CARD_IDS = frozenset({"STRIKE", "DEFEND", "BASH", "BLOODLETTING", "INFLAME"})


def _starter_only_deck(ctx: DeckContext) -> bool:
    return set(ctx.effects.keys()) <= _STARTER_CARD_IDS


def _max_single_hit_damage(st: GeneralState, ctx: DeckContext) -> int:
    str_p = st.strength_p
    inf = ctx.effects.get("INFLAME")
    if inf and inf.strength_apply > 0 and card_remaining(st, "INFLAME") > 0:
        str_p += inf.strength_apply
    vuln = st.vuln_e
    best = 0
    for cid, eff in ctx.effects.items():
        if eff.damage <= 0 or card_remaining(st, cid) <= 0:
            continue
        per_hit = calc_damage(eff.damage, str_p, vuln, 0)
        best = max(best, per_hit * max(1, eff.hits))
    return best


def _general_fight_key(st: GeneralState, ctx: DeckContext) -> tuple:
    block_e = st.block_e
    if block_e > 0:
        cap = _max_single_hit_damage(st, ctx)
        if cap > 0:
            block_e = min(block_e, cap)
    return (
        st.hp_p,
        st.block_p,
        st.strength_p,
        st.hp_e,
        block_e,
        st.vuln_e,
        st.pattern_idx,
        st.hand,
        st.draw,
        st.discard,
        st.turn,
        st.shuffles,
    )


def fight_key(st: GeneralState, ctx: DeckContext) -> tuple:
    if _starter_only_deck(ctx):
        return to_sim3_state(st).key()
    return _general_fight_key(st, ctx)


def _apply_forward_prunes(st: GeneralState, params: FightParams, ctx: DeckContext) -> str | None:
    if not _starter_only_deck(ctx):
        if st.hp_e <= 0:
            return None
        if prune_kill_shell_turn_budget(
            st.hp_e, st.turn, params.max_turns, params.enemy_hp_loss_cap
        ):
            return "kill"
        return None
    return sim3_dp._apply_forward_prunes(to_sim3_state(st), params)


def _ceiling_prune(
    st: GeneralState, params: FightParams, ctx: DeckContext, best_known_hp: int
) -> bool:
    if not _starter_only_deck(ctx):
        return False
    return sim3_dp._ceiling_prune(to_sim3_state(st), params, best_known_hp)


def _hand_has_playable_damage(st: GeneralState, ctx: DeckContext) -> bool:
    for cid, eff in ctx.effects.items():
        if eff.damage > 0 and st.hand_dict().get(cid, 0) > 0:
            return True
    return False


def _hand_has_playable_bl_or_inf(st: GeneralState) -> bool:
    h = st.hand_dict()
    return h.get("BLOODLETTING", 0) > 0 or h.get("INFLAME", 0) > 0


def ordered_plays(
    st: GeneralState,
    pattern: Pattern,
    ctx: DeckContext,
) -> list[PlayCounts]:
    hand = st.hand_dict()
    plays = legal_plays(hand, ctx.effects, base_energy=ctx.base_energy)

    incoming = 0
    if pattern:
        kind, val, _ = parse_pattern_step(pattern[st.pattern_idx % len(pattern)])
        if kind == "A":
            incoming = val

    filtered: list[PlayCounts] = []
    for play in plays:
        loss = play_hp_loss(play, ctx.effects)
        if loss > 0 and st.hp_p - loss <= 0:
            continue
        if (
            cards_played(play) == 0
            and _hand_has_playable_damage(st, ctx)
            and st.hp_e > 0
        ):
            continue
        if (
            play_has_block(play, ctx.effects)
            and incoming == 0
            and not play_has_damage(play, ctx.effects)
        ):
            if not play.get("BLOODLETTING") and not play.get("INFLAME"):
                continue
        if cards_played(play) == 0 and st.hp_e > 0:
            h = st.hand_dict()
            if (
                _hand_has_playable_damage(st, ctx)
                or _hand_has_playable_bl_or_inf(st)
                or (h.get("DEFEND", 0) > 0 and incoming > 0)
            ):
                continue
        filtered.append(play)

    def sort_key(play: PlayCounts) -> tuple:
        defends = play.get("DEFEND", 0)
        bl = play.get("BLOODLETTING", 0)
        inf = play.get("INFLAME", 0)
        bash = play.get("BASH", 0)
        strikes = play.get("STRIKE", 0)
        return (-defends, -bl, -inf, -bash, -strikes, -cards_played(play))

    filtered.sort(key=sort_key)
    return filtered or [{}]


PlayTuple = tuple[tuple[str, int], ...]
MemoEntry = tuple[DpValue, PlayTuple | None]


def _play_tuple(play: PlayCounts) -> PlayTuple:
    return tuple(sorted((k, v) for k, v in play.items() if v > 0))


def _child_value(
    st: GeneralState,
    play: PlayCounts,
    params: FightParams,
    shuffle_seed: int,
    ctx: DeckContext,
    memo: dict[tuple, MemoEntry],
    best_known: list[int],
    use_prunes: bool,
) -> DpValue:
    nxt = apply_turn(
        st,
        play,
        params.pattern,
        shuffle_seed,
        ctx.effects,
        enemy_slow=params.enemy_slow,
        enemy_hp_loss_cap=params.enemy_hp_loss_cap,
    )
    if nxt is None:
        return DpValue(LOSS_HP, 1)
    if nxt.hp_e <= 0:
        return DpValue(nxt.hp_p, 1)
    sub, _ = _solve_core(
        nxt, params, shuffle_seed, ctx, memo, best_known, use_prunes, share_play_bound=True
    )
    return DpValue(sub.final_hp, 1 + sub.turns_to_end)


def _solve_core(
    st: GeneralState,
    params: FightParams,
    shuffle_seed: int,
    ctx: DeckContext,
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

    key = fight_key(st, ctx)
    if key in memo:
        return memo[key]

    if use_prunes:
        reason = _apply_forward_prunes(st, params, ctx)
        if reason in ("survival", "kill"):
            entry: MemoEntry = (DpValue(LOSS_HP, 0), ())
            memo[key] = entry
            return entry
        if _ceiling_prune(st, params, ctx, best_known[0]):
            return DpValue(LOSS_HP, 0), None

    best = DpValue(LOSS_HP, 9999)
    best_play: PlayTuple | None = None
    for play in ordered_plays(st, params.pattern, ctx):
        bound = best_known if share_play_bound else [LOSS_HP]
        child = _child_value(
            st, play, params, shuffle_seed, ctx, memo, bound, use_prunes
        )
        if _rank(child) > _rank(best):
            best = child
            best_play = _play_tuple(play)
        if best.winnable and best.final_hp > best_known[0]:
            best_known[0] = best.final_hp

    entry = (best, best_play)
    memo[key] = entry
    return entry


def _rank(v: DpValue) -> tuple:
    return (v.winnable, v.final_hp, -v.turns_to_end)


def solve_tuple_general(
    st: GeneralState,
    params: FightParams,
    shuffle_seed: int,
    ctx: DeckContext,
    *,
    memo: dict[tuple, MemoEntry] | None = None,
    use_prunes: bool = True,
    best_known: list[int] | None = None,
) -> DpValue:
    table: dict[tuple, MemoEntry] = memo if memo is not None else {}
    bound = best_known if best_known is not None else [LOSS_HP]
    return _solve_core(st, params, shuffle_seed, ctx, table, bound, use_prunes)[0]


def solve_with_best_play_general(
    st: GeneralState,
    params: FightParams,
    shuffle_seed: int,
    ctx: DeckContext,
    *,
    memo: dict[tuple, MemoEntry] | None = None,
    use_prunes: bool = True,
    best_known: list[int] | None = None,
) -> MemoEntry:
    table: dict[tuple, MemoEntry] = memo if memo is not None else {}
    bound = best_known if best_known is not None else [LOSS_HP]
    return _solve_core(st, params, shuffle_seed, ctx, table, bound, use_prunes)
