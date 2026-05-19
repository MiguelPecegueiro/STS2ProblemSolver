"""General DP must match Sim 3 on the same fights."""

from __future__ import annotations

from combat_sim.card_effect import IRONCLAD_STARTER_EFFECTS
from combat_sim.general.plays import legal_plays, play_energy_ok, play_order
from combat_sim.general.state import DeckContext
from combat_sim.general.optimal import solve_optimal_general
from combat_sim.scenarios import bygone_effigy_sim3, jaw_worm_sim3, skulking_colony_sim3
from combat_sim.sim3.composition import feasible_plays_sim3
from combat_sim.sim3.optimal import solve_optimal_sim3
from combat_sim.sim3.tuple_dp import state_from_engine
from combat_sim.sim3.optimal import fight_params_from_state
from combat_sim.general.state import state_from_engine as general_state_from_engine


STARTER_CTX = DeckContext.from_effects(IRONCLAD_STARTER_EFFECTS)


def test_play_energy_matches_sim3_feasible() -> None:
    """Every Sim3 composition has an equivalent legal general play."""
    hand = {"STRIKE": 2, "DEFEND": 2, "BASH": 1}
    general = legal_plays(hand, STARTER_CTX.effects, base_energy=3)
    sim3 = feasible_plays_sim3(2, 2, 1, 0, 0)
    for comp in sim3:
        play = {
            "STRIKE": comp.strikes,
            "DEFEND": comp.defends,
            "BASH": comp.bash,
            "BLOODLETTING": comp.bloodletting,
            "INFLAME": comp.inflame,
        }
        assert play_energy_ok(play, STARTER_CTX.effects)
        assert {k: v for k, v in play.items() if v} in [
            {k: v for k, v in g.items() if v} for g in general
        ]


def test_play_order_bloodletting_before_strike() -> None:
    play = {"BLOODLETTING": 1, "STRIKE": 2}
    order = play_order(play, STARTER_CTX.effects)
    assert order.index("BLOODLETTING") < order.index("STRIKE")


def test_play_order_bash_before_strike_for_vuln() -> None:
    play = {"BASH": 1, "STRIKE": 1}
    order = play_order(play, STARTER_CTX.effects)
    assert order.index("BASH") < order.index("STRIKE")


def test_general_dp_matches_sim3_jaw_worm() -> None:
    state = jaw_worm_sim3(seed=42)
    sim3 = solve_optimal_sim3(state, shuffle_seed=42, max_turns=20)
    general = solve_optimal_general(state, shuffle_seed=42, max_turns=20)
    assert general.final_hp == sim3.final_hp
    assert general.turns_to_end == sim3.turns_to_end
    assert general.winnable == sim3.winnable


def test_general_dp_matches_sim3_bygone_effigy() -> None:
    state = bygone_effigy_sim3(seed=42)
    sim3 = solve_optimal_sim3(state, shuffle_seed=42, max_turns=25)
    general = solve_optimal_general(state, shuffle_seed=42, max_turns=25)
    assert general.final_hp == sim3.final_hp
    assert general.turns_to_end == sim3.turns_to_end


def test_general_dp_matches_sim3_skulking_colony() -> None:
    state = skulking_colony_sim3(seed=42)
    sim3 = solve_optimal_sim3(state, shuffle_seed=42, max_turns=25)
    general = solve_optimal_general(state, shuffle_seed=42, max_turns=25)
    assert general.final_hp == sim3.final_hp
    assert general.turns_to_end == sim3.turns_to_end


def test_state_from_engine_hand_counts_match() -> None:
    state = jaw_worm_sim3(seed=7)
    params = fight_params_from_state(state, max_turns=20)
    s3 = state_from_engine(state, params.pattern)
    gen = general_state_from_engine(state, params.pattern, STARTER_CTX)
    assert gen.hand_dict().get("STRIKE", 0) == s3.hand_s
    assert gen.hand_dict().get("DEFEND", 0) == s3.hand_d
    assert gen.hand_dict().get("BASH", 0) == s3.hand_b
