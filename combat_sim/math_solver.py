"""Monte Carlo math solver: win probability + composition-optimal turns."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from combat_sim.deck_counts import hand_counts
from combat_sim.engine import CombatEngine, TurnAction
from combat_sim.solver import solve_turn
from combat_sim.state import CardInstance, CombatPhase, CombatState


@dataclass
class MathSolverConfig:
    """Tuning for decision quality vs speed."""

    decision_rollouts: int = 40
    estimate_rollouts: int = 120
    max_turns: int = 30
    seed: int | None = None

    def fast_batch(self) -> MathSolverConfig:
        return MathSolverConfig(
            decision_rollouts=8,
            estimate_rollouts=12,
            max_turns=self.max_turns,
            seed=self.seed,
        )


@dataclass(frozen=True, slots=True)
class TurnComposition:
    strikes: int
    defends: int

    @property
    def cards_played(self) -> int:
        return self.strikes + self.defends

    def label(self) -> str:
        if self.cards_played == 0:
            return "End Turn (no plays)"
        return f"{self.strikes} Strike + {self.defends} Defend"


@dataclass
class TurnDecision:
    action: TurnAction
    composition: TurnComposition
    win_probability: float
    candidates: list[tuple[TurnComposition, float]] = field(default_factory=list)


def estimate_win_probability(
    state: CombatState,
    config: MathSolverConfig,
    *,
    rng: random.Random | None = None,
) -> float:
    """Monte Carlo P(win) from this state to fight end."""
    if state.phase == CombatPhase.WON:
        return 1.0
    if state.phase == CombatPhase.LOST:
        return 0.0
    if state.phase != CombatPhase.PLAYER:
        return 0.0

    rng = rng or _rng_for(config, salt=0)
    n = max(1, config.estimate_rollouts)
    wins = 0
    for i in range(n):
        sim = state.copy()
        sim.rng = random.Random(rng.randint(0, 2**31 - 1))
        if _rollout_to_end(sim, config, sim.rng):
            wins += 1
    return wins / n


def choose_turn(state: CombatState, config: MathSolverConfig) -> TurnDecision:
    """Pick (strikes, defends) to play that maximizes estimated win rate."""
    if state.phase != CombatPhase.PLAYER:
        return TurnDecision(TurnAction(), TurnComposition(0, 0), _terminal_win_prob(state))

    rng = _rng_for(config, salt=state.turn * 1000 + state.player_hp)
    candidates: list[tuple[TurnComposition, float, tuple]] = []

    for comp in feasible_compositions(state):
        trial = state.copy()
        hp_before = trial.player_hp
        enemy_hp_before = sum(e.hp for e in trial.enemies)
        _apply_plays_only(trial, action_for_composition(trial, comp))
        if trial.phase == CombatPhase.WON:
            p_win = 1.0
            hp_lost = 0
            damage = enemy_hp_before
        elif trial.phase == CombatPhase.LOST:
            p_win = 0.0
            hp_lost = hp_before
            damage = enemy_hp_before - sum(e.hp for e in trial.enemies)
        elif trial.phase == CombatPhase.PLAYER:
            CombatEngine.end_turn(trial)
            hp_lost = max(0, hp_before - trial.player_hp)
            damage = enemy_hp_before - sum(e.hp for e in trial.enemies)
            if trial.phase == CombatPhase.WON:
                p_win = 1.0
            elif trial.phase == CombatPhase.LOST:
                p_win = 0.0
            else:
                eval_cfg = MathSolverConfig(
                    decision_rollouts=config.decision_rollouts,
                    estimate_rollouts=config.decision_rollouts,
                    max_turns=config.max_turns,
                    seed=rng.randint(0, 2**31 - 1),
                )
                p_win = estimate_win_probability(trial, eval_cfg, rng=rng)
        else:
            p_win = 0.0
            hp_lost = hp_before
            damage = 0

        tie = (p_win, -hp_lost, damage, comp.cards_played)
        candidates.append((comp, p_win, tie))

    best_comp, best_p, _ = max(candidates, key=lambda x: x[2])
    ranked = [(c, p) for c, p, _ in sorted(candidates, key=lambda x: -x[2][0])]
    return TurnDecision(
        action=action_for_composition(state, best_comp),
        composition=best_comp,
        win_probability=best_p,
        candidates=ranked[:8],
    )


def _terminal_win_prob(state: CombatState) -> float:
    if state.phase == CombatPhase.WON:
        return 1.0
    if state.phase == CombatPhase.LOST:
        return 0.0
    return 0.0


def _rollout_to_end(
    state: CombatState,
    config: MathSolverConfig,
    rng: random.Random,
) -> bool:
    """Fast playout to terminal; always uses greedy :func:`solve_turn`."""
    del rng
    turns = 0
    while state.phase == CombatPhase.PLAYER and turns < config.max_turns:
        if state.all_enemies_dead():
            state.phase = CombatPhase.WON
            break
        CombatEngine.apply_turn(state, solve_turn(state))
        turns += 1
        if state.player_hp <= 0:
            state.phase = CombatPhase.LOST
            break
    return state.phase == CombatPhase.WON


def feasible_compositions(state: CombatState) -> list[TurnComposition]:
    hc = hand_counts(state)
    max_play = min(3, state.energy, hc.total)
    out: list[TurnComposition] = []
    for total in range(0, max_play + 1):
        for strikes in range(0, min(total, hc.strikes) + 1):
            defends = total - strikes
            if defends <= hc.defends:
                out.append(TurnComposition(strikes, defends))
    return out


def action_for_composition(state: CombatState, comp: TurnComposition) -> TurnAction:
    living = state.living_enemies()
    if not living and comp.strikes > 0:
        return TurnAction()
    target = living[0].enemy_id if living else None

    defends = [c for c in state.hand if c.definition.block > 0]
    strikes = [c for c in state.hand if c.definition.damage > 0]
    plays: list[tuple[int, str | None]] = []
    for card in defends[: comp.defends]:
        plays.append((card.instance_id, None))
    for card in strikes[: comp.strikes]:
        plays.append((card.instance_id, target))
    return TurnAction(tuple(plays))


def _apply_plays_only(state: CombatState, action: TurnAction) -> None:
    for instance_id, target_id in action.plays:
        if not CombatEngine.play_card(state, instance_id, target_id):
            break


def _rng_for(config: MathSolverConfig, *, salt: int) -> random.Random:
    base = config.seed if config.seed is not None else 0
    return random.Random(base + salt)
