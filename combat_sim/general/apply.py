"""Apply a turn of card plays on scalar combat state (tuple DP)."""

from __future__ import annotations

from combat_sim.card_effect import CardEffect
from combat_sim.damage_util import apply_damage_to_enemy_hp_block, calc_damage
from combat_sim.general.hand import HandCounts, PlayCounts, counts_to_tuple, subtract_hand
from combat_sim.general.pile import shuffle_discard_into_draw, sort_discard_pile
from combat_sim.general.plays import play_order
from combat_sim.general.state import GeneralState
from combat_sim.pattern_util import parse_pattern_step
from combat_sim.sim3.tuple_dp import Pattern
from combat_sim.tuple_dp import HAND_DRAW


def apply_player_plays(
    hp_p: int,
    hp_e: int,
    block_e: int,
    vuln_e: int,
    strength_p: int,
    play: PlayCounts,
    effects: dict[str, CardEffect],
    *,
    enemy_slow: bool = False,
    enemy_hp_loss_cap: int | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Resolve player cards in play_order; return (hp_p, hp_e, block_e, vuln_e, strength_p, block_p)."""
    slow_n = 0
    hp_lost = 0
    block_p = 0

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

    for cid in play_order(play, effects):
        eff = effects[cid]
        if eff.hp_loss > 0:
            hp_p -= eff.hp_loss
            if enemy_slow:
                slow_n += 1
        if eff.block > 0:
            block_p += eff.block
            if enemy_slow:
                slow_n += 1
        if eff.strength_apply > 0:
            strength_p += eff.strength_apply
            if enemy_slow:
                slow_n += 1
        for _ in range(max(1, eff.hits)):
            if eff.damage > 0:
                slow = slow_n if enemy_slow else 0
                dmg = calc_damage(eff.damage, strength_p, vuln_e, slow)
                _hit(dmg)
                if enemy_slow:
                    slow_n += 1
        if eff.vuln_apply > 0:
            vuln_e += eff.vuln_apply

    return hp_p, hp_e, block_e, vuln_e, strength_p, block_p


def apply_turn(
    st: GeneralState,
    play: PlayCounts,
    pattern: Pattern,
    shuffle_seed: int,
    effects: dict[str, CardEffect],
    *,
    enemy_slow: bool = False,
    enemy_hp_loss_cap: int | None = None,
) -> GeneralState | None:
    hand = st.hand_dict()
    for cid, n in play.items():
        if n > hand.get(cid, 0):
            return None
    if not all(n >= 0 for n in play.values()):
        return None

    hp_p, hp_e, block_e, vuln_e, strength_p, block_p = apply_player_plays(
        st.hp_p,
        st.hp_e,
        st.block_e,
        st.vuln_e,
        st.strength_p,
        play,
        effects,
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
        return GeneralState(
            hp_p=hp_p,
            strength_p=strength_p,
            hp_e=0,
            block_e=block_e,
            vuln_e=vuln_e,
            pattern_idx=idx,
            hand=(),
            draw=st.draw,
            discard=st.discard,
            turn=st.turn,
            shuffles=st.shuffles,
        )

    remaining = subtract_hand(hand, play)
    played_ids = play_order(play, effects)
    discard_list = list(st.discard)
    for cid in played_ids:
        if effects[cid].exhaust:
            continue
        discard_list.append(cid)
    for cid, n in sorted(remaining.items()):
        discard_list.extend([cid] * n)
    discard = sort_discard_pile(tuple(discard_list))

    draw = st.draw
    new_hand: HandCounts = {}
    shuffles = st.shuffles
    for _ in range(HAND_DRAW):
        if not draw:
            if not discard:
                break
            draw = shuffle_discard_into_draw(discard, shuffle_seed, shuffles)
            shuffles += 1
            discard = ()
        if draw:
            card = draw[-1]
            draw = draw[:-1]
            new_hand[card] = new_hand.get(card, 0) + 1

    vuln_e = max(0, vuln_e - 1)

    return GeneralState(
        hp_p=hp_p,
        strength_p=strength_p,
        hp_e=hp_e,
        block_e=block_e,
        vuln_e=vuln_e,
        pattern_idx=idx,
        hand=counts_to_tuple(new_hand),
        draw=draw,
        discard=discard,
        turn=st.turn + 1,
        shuffles=shuffles,
    )
