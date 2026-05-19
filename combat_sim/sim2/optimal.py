"""Bridge engine CombatState <-> Sim 2 tuple DP."""

from __future__ import annotations

from dataclasses import dataclass, field

from combat_sim.engine import TurnAction
from combat_sim.sim2.composition import TurnCompositionSim2
from combat_sim.sim2.tuple_dp import (
    DpValue,
    FightParams,
    LOSS_HP,
    _child_value,
    _rank,
    ordered_plays_sim2,
    solve_with_best_play_sim2,
    state_from_engine,
)
from combat_sim.pattern_util import pattern_from_intents
from combat_sim.state import CombatState


def fight_params_from_state(state: CombatState, *, max_turns: int = 30) -> FightParams:
    enemy = state.enemies[0]
    return FightParams(
        player_hp=state.player_max_hp,
        enemy_hp=enemy.max_hp,
        pattern=pattern_from_intents(enemy.pattern),
        max_turns=max_turns,
        enemy_slow=enemy.has_slow,
    )


def solve_optimal_sim2(
    state: CombatState,
    *,
    max_turns: int = 30,
    shuffle_seed: int = 0,
    memo: dict | None = None,
) -> DpValue:
    params = fight_params_from_state(state, max_turns=max_turns)
    st = state_from_engine(state, params.pattern)
    return solve_with_best_play_sim2(st, params, shuffle_seed, memo=memo)[0]


@dataclass
class OptimalTurnDecisionSim2:
    action: TurnAction
    composition: TurnCompositionSim2
    value: DpValue
    candidates: list[tuple[TurnCompositionSim2, DpValue]] = field(default_factory=list)

    @property
    def win_probability(self) -> float:
        return 1.0 if self.value.winnable else 0.0


def action_for_composition_sim2(
    state: CombatState,
    comp: TurnCompositionSim2,
) -> TurnAction:
    living = state.living_enemies()
    target = living[0].enemy_id if living else None

    bloodlettings = [c for c in state.hand if c.definition.card_id == "BLOODLETTING"]
    defends = [c for c in state.hand if c.definition.block > 0 and c.definition.card_id != "BASH"]
    bashes = [c for c in state.hand if c.definition.card_id == "BASH"]
    strikes = [c for c in state.hand if c.definition.card_id == "STRIKE"]

    plays: list[tuple[int, str | None]] = []
    for card in bloodlettings[: comp.bloodletting]:
        plays.append((card.instance_id, None))
    for card in defends[: comp.defends]:
        plays.append((card.instance_id, None))
    for card in bashes[: comp.bash]:
        plays.append((card.instance_id, target))
    for card in strikes[: comp.strikes]:
        plays.append((card.instance_id, target))
    return TurnAction(tuple(plays))


def choose_optimal_turn_sim2(
    state: CombatState,
    *,
    max_turns: int = 30,
    shuffle_seed: int = 0,
    list_candidates: bool = True,
    memo: dict | None = None,
) -> OptimalTurnDecisionSim2:
    params = fight_params_from_state(state, max_turns=max_turns)
    st = state_from_engine(state, params.pattern)
    table: dict = memo if memo is not None else {}
    best_known = [LOSS_HP]

    value, best_comp = solve_with_best_play_sim2(
        st, params, shuffle_seed, memo=table, use_prunes=True, best_known=best_known
    )
    comp = best_comp or TurnCompositionSim2(0, 0, 0, 0)

    candidates: list[tuple[TurnCompositionSim2, DpValue]] = []
    if list_candidates:
        for c in ordered_plays_sim2(st, params.pattern):
            child = _child_value(st, c, params, shuffle_seed, table, [LOSS_HP], use_prunes=True)
            candidates.append((c, child))
        candidates.sort(key=lambda x: _rank(x[1]), reverse=True)

    return OptimalTurnDecisionSim2(
        action=action_for_composition_sim2(state, comp),
        composition=comp,
        value=value,
        candidates=candidates,
    )
