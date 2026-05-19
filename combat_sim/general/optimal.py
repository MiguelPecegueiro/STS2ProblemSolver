"""Bridge CombatState <-> general tuple DP."""

from __future__ import annotations

from combat_sim.card_effect import IRONCLAD_STARTER_EFFECTS
from combat_sim.general.state import DeckContext, GeneralState, state_from_engine
from combat_sim.general.tuple_dp import DpValue, FightParams, solve_with_best_play_general
from combat_sim.pattern_util import pattern_from_intents
from combat_sim.sim3.optimal import fight_params_from_state
from combat_sim.state import CombatState

STARTER_DECK_EFFECTS = IRONCLAD_STARTER_EFFECTS
STARTER_CTX = DeckContext.from_effects(STARTER_DECK_EFFECTS)


def solve_optimal_general(
    state: CombatState,
    *,
    deck_effects: tuple = STARTER_DECK_EFFECTS,
    max_turns: int = 30,
    shuffle_seed: int = 0,
    memo: dict | None = None,
) -> DpValue:
    ctx = DeckContext.from_effects(deck_effects)
    params = fight_params_from_state(state, max_turns=max_turns)
    st = state_from_engine(state, params.pattern, ctx)
    return solve_with_best_play_general(st, params, shuffle_seed, ctx, memo=memo)[0]


def solve_general(
    deck_effects: tuple,
    params: FightParams,
    shuffle_seed: int,
    st: GeneralState,
) -> DpValue:
    ctx = DeckContext.from_effects(deck_effects)
    return solve_with_best_play_general(st, params, shuffle_seed, ctx)[0]
