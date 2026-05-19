"""Twin Strike — multi-hit validation on the general tuple DP."""

from __future__ import annotations

from combat_sim.card_effect import (
    IRONCLAD_STARTER_SIM3_TWIN_STRIKE_EFFECTS,
    TWIN_STRIKE_EFFECT,
)
from combat_sim.damage_util import calc_damage
from combat_sim.general.apply import apply_player_plays
from combat_sim.general.optimal import solve_optimal_general
from combat_sim.general.plays import play_order
from combat_sim.general.state import DeckContext
from combat_sim.scenarios import twin_strike_jaw_worm_sim3

TWIN_CTX = DeckContext.from_effects(IRONCLAD_STARTER_SIM3_TWIN_STRIKE_EFFECTS)


def test_twin_strike_play_order_after_block() -> None:
    play = {"DEFEND": 1, "TWIN_STRIKE": 1}
    order = play_order(play, TWIN_CTX.effects)
    assert order.index("DEFEND") < order.index("TWIN_STRIKE")


def test_twin_strike_slow_two_hits_5_and_6() -> None:
    """With Slow 1 before Twin Strike: floor(5*1.1) + floor(5*1.2) = 11."""
    play = {"DEFEND": 1, "TWIN_STRIKE": 1}
    hp_p, hp_e, block_e, vuln_e, strength_p, block_p = apply_player_plays(
        80,
        127,
        0,
        0,
        0,
        play,
        TWIN_CTX.effects,
        enemy_slow=True,
    )
    assert block_p == 5
    assert hp_e == 127 - 11
    hit1 = calc_damage(5, 0, 0, 1)
    hit2 = calc_damage(5, 0, 0, 2)
    assert hit1 == 5
    assert hit2 == 6
    assert hit1 + hit2 == 11


def test_twin_strike_vuln_applies_to_each_hit() -> None:
    """Pre-existing vuln multiplies both hits independently."""
    play = {"TWIN_STRIKE": 1}
    _, hp_e, _, _, _, _ = apply_player_plays(
        80,
        40,
        0,
        2,
        0,
        play,
        TWIN_CTX.effects,
        enemy_slow=False,
    )
    per_hit = calc_damage(5, 0, 2, 0)
    assert per_hit == 7
    assert hp_e == 40 - 2 * per_hit


def test_twin_strike_effigy_turn1_damage_matches_card_engine() -> None:
    """General tuple apply matches apply_card_effect (multi-hit card engine)."""
    from combat_sim.card_effect import BLOODLETTING_EFFECT
    from combat_sim.card_engine import apply_card_effect
    from combat_sim.scenarios import _bygone_effigy_enemy
    from combat_sim.state import CombatState

    state = CombatState.new_fight(
        deck=[],
        player_hp=80,
        enemies=[_bygone_effigy_enemy()],
        seed=0,
    )
    eid = state.enemies[0].enemy_id
    apply_card_effect(state, BLOODLETTING_EFFECT, None)
    apply_card_effect(state, TWIN_STRIKE_EFFECT, eid)
    engine_hp = state.enemies[0].hp

    play = {"BLOODLETTING": 1, "TWIN_STRIKE": 1}
    _, hp_e, _, _, _, _ = apply_player_plays(
        80,
        127,
        0,
        0,
        0,
        play,
        TWIN_CTX.effects,
        enemy_slow=True,
    )
    assert hp_e == engine_hp


def test_twin_strike_jaw_worm_general_dp_seed42() -> None:
    state = twin_strike_jaw_worm_sim3(seed=42)
    val = solve_optimal_general(
        state,
        deck_effects=IRONCLAD_STARTER_SIM3_TWIN_STRIKE_EFFECTS,
        shuffle_seed=42,
        max_turns=20,
    )
    assert val.winnable
    assert val.final_hp > 0


def test_twin_strike_deck_has_13_cards() -> None:
    state = twin_strike_jaw_worm_sim3(seed=42)
    assert len(state.draw_pile) + len(state.hand) == 13


def test_twin_strike_legal_play_in_hand() -> None:
    from combat_sim.general.plays import legal_plays

    plays = legal_plays({"TWIN_STRIKE": 1}, TWIN_CTX.effects)
    assert {"TWIN_STRIKE": 1} in [{k: v for k, v in p.items() if v} for p in plays]
