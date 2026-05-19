"""Exact DP solver (tuple core backend)."""

from __future__ import annotations

from dataclasses import dataclass, field
from combat_sim.math_solver import TurnComposition, action_for_composition
from combat_sim.state import CombatPhase, CombatState, IntentKind
from combat_sim.tuple_dp import (
    DpValue,
    FightParams,
    Pattern,
    solve_with_best_play,
    solve_tuple,
    state_from_engine,
)

LOSS_HP = -1_000_000_000

# Re-export for callers
__all__ = [
    "DpValue",
    "LOSS_HP",
    "OptimalTurnDecision",
    "choose_optimal_turn",
    "fight_params_from_state",
    "solve_optimal",
]


@dataclass
class _DpEntry:
    value: DpValue
    best_comp: TurnComposition | None = None


@dataclass
class OptimalTurnDecision:
    action: TurnAction
    composition: TurnComposition
    value: DpValue
    candidates: list[tuple[TurnComposition, DpValue]] = field(default_factory=list)

    @property
    def win_probability(self) -> float:
        return 1.0 if self.value.winnable else 0.0


def fight_params_from_state(state: CombatState, *, max_turns: int = 30) -> FightParams:
    enemy = state.enemies[0]
    pattern: Pattern = tuple(
        ("A", intent.value)
        if intent.kind == IntentKind.ATTACK
        else ("B", intent.value)
        for intent in enemy.pattern
    )
    return FightParams(
        player_hp=state.player_max_hp,
        enemy_hp=enemy.max_hp,
        pattern=pattern,
        max_turns=max_turns,
    )


def solve_optimal(
    state: CombatState,
    *,
    max_turns: int = 30,
    shuffle_seed: int = 0,
    memo: dict | None = None,
) -> DpValue:
    params = fight_params_from_state(state, max_turns=max_turns)
    st = state_from_engine(state, params.pattern)
    table: dict = memo if memo is not None else {}
    return solve_tuple(st, params, shuffle_seed)


def choose_optimal_turn(
    state: CombatState,
    *,
    max_turns: int = 30,
    shuffle_seed: int = 0,
    list_candidates: bool = True,
    memo: dict | None = None,
) -> OptimalTurnDecision:
    params = fight_params_from_state(state, max_turns=max_turns)
    st = state_from_engine(state, params.pattern)
    table: dict = memo if memo is not None else {}

    value, best_comp = solve_with_best_play(st, params, shuffle_seed, memo=table)
    comp = best_comp or TurnComposition(0, 0)

    candidates: list[tuple[TurnComposition, DpValue]] = []
    if list_candidates:
        from combat_sim.tuple_dp import apply_turn, ordered_plays

        for c in ordered_plays(st, params.pattern):
            nxt = apply_turn(st, c, params.pattern, shuffle_seed)
            if nxt is None:
                child = DpValue(LOSS_HP, 1)
            elif nxt.hp_e <= 0:
                child = DpValue(nxt.hp_p, 1)
            else:
                child_val = solve_tuple(nxt, params, shuffle_seed)
                child = DpValue(child_val.final_hp, 1 + child_val.turns_to_end)
            candidates.append((c, child))
        candidates.sort(key=lambda x: (_rank(x[1]),), reverse=True)

    return OptimalTurnDecision(
        action=action_for_composition(state, comp),
        composition=comp,
        value=value,
        candidates=candidates,
    )


def _rank(value: DpValue) -> tuple:
    return (value.winnable, value.final_hp, -value.turns_to_end)
