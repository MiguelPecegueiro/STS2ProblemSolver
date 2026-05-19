"""General card-id tuple DP — deck defined by CardEffect data."""

from combat_sim.general.optimal import solve_optimal_general
from combat_sim.general.plays import legal_plays, play_energy_ok, play_order
from combat_sim.general.state import DeckContext, GeneralState, state_from_engine
from combat_sim.general.tuple_dp import DpValue, FightParams, solve_tuple_general

__all__ = [
    "DeckContext",
    "DpValue",
    "FightParams",
    "GeneralState",
    "legal_plays",
    "play_energy_ok",
    "play_order",
    "solve_optimal_general",
    "solve_tuple_general",
    "state_from_engine",
]
