"""Run combat fights with logging or batch reports (optimal DP or Monte Carlo)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from combat_sim.deck_counts import deck_totals, format_hand_counts
from combat_sim.engine import CombatEngine
from combat_sim.math_solver import MathSolverConfig, TurnDecision, choose_turn, estimate_win_probability
from combat_sim.solver import solve_turn
from combat_sim.optimal_solver import OptimalTurnDecision, choose_optimal_turn
from combat_sim.state import CombatPhase, CombatState

ScenarioFn = Callable[..., CombatState]
SolverMode = Literal["optimal", "mc", "greedy"]


def _choose_optimal(state, *, max_turns, shuffle_seed, list_candidates, memo, sim: int):
    if sim >= 3:
        from combat_sim.sim3.optimal import choose_optimal_turn_sim3

        return choose_optimal_turn_sim3(
            state,
            max_turns=max_turns,
            shuffle_seed=shuffle_seed,
            list_candidates=list_candidates,
            memo=memo,
        )
    if sim >= 2:
        from combat_sim.sim2.optimal import choose_optimal_turn_sim2

        return choose_optimal_turn_sim2(
            state,
            max_turns=max_turns,
            shuffle_seed=shuffle_seed,
            list_candidates=list_candidates,
            memo=memo,
        )
    if sim >= 1:
        from combat_sim.sim1.optimal import choose_optimal_turn_sim1

        return choose_optimal_turn_sim1(
            state,
            max_turns=max_turns,
            shuffle_seed=shuffle_seed,
            list_candidates=list_candidates,
            memo=memo,
        )
    return choose_optimal_turn(
        state,
        max_turns=max_turns,
        shuffle_seed=shuffle_seed,
        list_candidates=list_candidates,
        memo=memo,
    )


@dataclass
class TurnLog:
    turn: int
    player_hp: int
    player_block: int
    energy: int
    hand_label: str
    deck_remaining: str
    enemy_summary: str
    incoming: str
    win_probability: float
    optimal_final_hp: int | None
    chosen: str
    candidate_lines: list[str] = field(default_factory=list)


@dataclass
class LoggedFight:
    scenario: str
    seed: int | None
    won: bool
    final_hp: int
    turns: int
    solver: str
    turn_logs: list[TurnLog] = field(default_factory=list)


@dataclass
class BatchReport:
    scenario: str
    runs: int
    wins: int
    win_rate: float
    avg_turns: float
    avg_final_hp: float
    avg_final_hp_wins: float
    avg_final_hp_losses: float
    avg_enemy_hp_remaining: float
    avg_enemy_hp_remaining_losses: float
    avg_enemy_hp_pct_remaining: float
    loss_turn_histogram: dict[int, int] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    config_summary: str = ""

    def format_report(self) -> str:
        lines = [
            "=" * 60,
            f"COMBAT SIM BATCH REPORT — {self.scenario}",
            "=" * 60,
            f"Runs:              {self.runs}",
            f"Wins:              {self.wins}",
            f"Win rate:          {self.win_rate * 100:.2f}%",
            f"Avg turns (all):   {self.avg_turns:.2f}",
            f"Avg player HP:     {self.avg_final_hp:.2f}",
            f"Avg HP (wins):     {self.avg_final_hp_wins:.2f}",
            f"Avg HP (losses):   {self.avg_final_hp_losses:.2f}",
            f"Avg enemy HP left: {self.avg_enemy_hp_remaining:.2f}",
            f"Avg enemy HP left (losses): {self.avg_enemy_hp_remaining_losses:.2f}",
            f"Avg enemy % HP left: {self.avg_enemy_hp_pct_remaining * 100:.2f}%",
            f"Elapsed:           {self.elapsed_sec:.3f}s",
            f"Solver:            {self.config_summary}",
            "",
            "Losses by turn (player died after enemy phase):",
        ]
        if not self.loss_turn_histogram:
            lines.append("  (none)")
        else:
            for turn in sorted(self.loss_turn_histogram):
                lines.append(f"  Turn {turn}: {self.loss_turn_histogram[turn]}")
        lines.append("=" * 60)
        return "\n".join(lines)


def run_logged_fight(
    scenario: ScenarioFn,
    *,
    scenario_name: str = "fight",
    seed: int | None = 42,
    solver: SolverMode = "optimal",
    config: MathSolverConfig | None = None,
    max_turns: int = 30,
    verbose: bool = True,
    sim: int = 0,
) -> LoggedFight:
    """Play one fight; log exact optimal line or MC estimates."""
    config = config or MathSolverConfig(seed=seed)
    state = scenario(seed=seed) if seed is not None else scenario()
    logs: list[TurnLog] = []

    if verbose:
        label = "Optimal DP" if solver == "optimal" else "Monte Carlo"
        sim_label = f" Sim {sim}" if sim else ""
        print(_header(scenario_name, seed, label + sim_label))
        print(_state_snapshot(state, sim=sim))
        print()

    dp_memo: dict = {} if solver == "optimal" else {}

    while state.phase == CombatPhase.PLAYER and state.turn <= max_turns:
        if state.all_enemies_dead():
            state.phase = CombatPhase.WON
            break

        if solver == "optimal":
            decision = _choose_optimal(
                state,
                max_turns=max_turns,
                shuffle_seed=seed if seed is not None else 0,
                list_candidates=verbose,
                memo=dp_memo,
                sim=sim,
            )
            tlog = _build_optimal_turn_log(state, decision, sim=sim)
            action = decision.action
        else:
            win_p = estimate_win_probability(state, config)
            mc_decision = choose_turn(state, config)
            tlog = _build_mc_turn_log(state, win_p, mc_decision)
            action = mc_decision.action

        logs.append(tlog)
        if verbose:
            print(_format_turn_log(tlog))

        CombatEngine.apply_turn(state, action)
        if state.player_hp <= 0:
            state.phase = CombatPhase.LOST

    result = LoggedFight(
        scenario=scenario_name,
        seed=seed,
        won=state.phase == CombatPhase.WON,
        final_hp=state.player_hp,
        turns=state.turn,
        solver=solver,
        turn_logs=logs,
    )

    if verbose:
        print()
        enemy_left = _enemy_hp_remaining(state)
        print(
            f"RESULT: {'WIN' if result.won else 'LOSS'} | HP={result.final_hp} | "
            f"turns={result.turns} | enemy HP left={enemy_left}"
        )
    return result


def run_batch(
    scenario: ScenarioFn,
    *,
    scenario_name: str = "fight",
    runs: int,
    solver: SolverMode = "optimal",
    config: MathSolverConfig | None = None,
    fast: bool = False,
    base_seed: int = 0,
    max_turns: int = 30,
    sim: int = 0,
) -> BatchReport:
    """Run many fights; aggregate win rate (exact per seed with optimal solver)."""
    config = config or MathSolverConfig()
    if fast and solver == "mc":
        config = config.fast_batch()

    wins = 0
    turns_sum = 0
    hp_sum = 0
    hp_wins: list[int] = []
    hp_losses: list[int] = []
    loss_turns: dict[int, int] = {}
    enemy_hp_sum = 0
    enemy_hp_losses: list[int] = []
    enemy_pct_sum = 0.0

    t0 = time.perf_counter()
    for i in range(runs):
        seed = base_seed + i
        state = scenario(seed=seed)
        fight_turns = 0
        while state.phase == CombatPhase.PLAYER and fight_turns < max_turns:
            if state.all_enemies_dead():
                state.phase = CombatPhase.WON
                break
            if solver == "optimal":
                action = _choose_optimal(
                    state,
                    max_turns=max_turns,
                    shuffle_seed=seed,
                    list_candidates=False,
                    memo={},
                    sim=sim,
                ).action
            elif solver == "greedy":
                action = solve_turn(state)
            else:
                fight_config = MathSolverConfig(
                    decision_rollouts=config.decision_rollouts,
                    estimate_rollouts=config.estimate_rollouts,
                    max_turns=max_turns,
                    seed=seed,
                )
                action = choose_turn(state, fight_config).action
            CombatEngine.apply_turn(state, action)
            fight_turns += 1
            if state.player_hp <= 0:
                state.phase = CombatPhase.LOST
                loss_turns[state.turn] = loss_turns.get(state.turn, 0) + 1
                break

        if state.phase == CombatPhase.WON:
            wins += 1
            hp_wins.append(state.player_hp)
        else:
            hp_losses.append(state.player_hp)
        turns_sum += state.turn
        hp_sum += state.player_hp

        enemy_hp_left = _enemy_hp_remaining(state)
        enemy_hp_sum += enemy_hp_left
        if state.phase != CombatPhase.WON:
            enemy_hp_losses.append(enemy_hp_left)
        enemy_pct_sum += _enemy_hp_fraction_remaining(state)

    elapsed = time.perf_counter() - t0
    n = max(runs, 1)
    if solver == "optimal":
        summary = f"optimal DP Sim {sim}, max_turns={max_turns}"
    elif solver == "greedy":
        summary = f"greedy turn solver, max_turns={max_turns}"
    else:
        summary = (
            f"MC decision_rollouts={config.decision_rollouts}, "
            f"estimate_rollouts={config.estimate_rollouts}, fast={fast}"
        )
    return BatchReport(
        scenario=scenario_name,
        runs=runs,
        wins=wins,
        win_rate=wins / n,
        avg_turns=turns_sum / n,
        avg_final_hp=hp_sum / n,
        avg_final_hp_wins=(sum(hp_wins) / len(hp_wins)) if hp_wins else 0.0,
        avg_final_hp_losses=(sum(hp_losses) / len(hp_losses)) if hp_losses else 0.0,
        avg_enemy_hp_remaining=enemy_hp_sum / n,
        avg_enemy_hp_remaining_losses=(
            (sum(enemy_hp_losses) / len(enemy_hp_losses)) if enemy_hp_losses else 0.0
        ),
        avg_enemy_hp_pct_remaining=enemy_pct_sum / n,
        loss_turn_histogram=loss_turns,
        elapsed_sec=elapsed,
        config_summary=summary,
    )


def _enemy_hp_remaining(state: CombatState) -> int:
    """Enemy HP at fight end (0 if killed)."""
    if not state.enemies:
        return 0
    return max(0, state.enemies[0].hp)


def _enemy_hp_fraction_remaining(state: CombatState) -> float:
    if not state.enemies:
        return 0.0
    enemy = state.enemies[0]
    if enemy.max_hp <= 0:
        return 0.0
    return max(0, enemy.hp) / enemy.max_hp


def _build_optimal_turn_log(state: CombatState, decision, *, sim: int = 0) -> TurnLog:
    enemy = state.living_enemies()[0] if state.living_enemies() else None
    intent = enemy.current_intent().label if enemy and enemy.current_intent() else "—"
    if enemy:
        enemy_summary = (
            f"{enemy.name} HP {enemy.hp}/{enemy.max_hp} block {enemy.block}"
            f" vuln {enemy.vuln_stacks}"
        )
    else:
        enemy_summary = "—"
    if sim >= 3:
        from combat_sim.deck_counts import deck_totals_sim3, format_hand_counts_sim3

        hand_label = format_hand_counts_sim3(state)
        deck_remaining = str(deck_totals_sim3(state))
    elif sim >= 2:
        from combat_sim.deck_counts import deck_totals_sim2, format_hand_counts_sim2

        hand_label = format_hand_counts_sim2(state)
        deck_remaining = str(deck_totals_sim2(state))
    elif sim >= 1:
        from combat_sim.deck_counts import deck_totals_sim1, format_hand_counts_sim1

        hand_label = format_hand_counts_sim1(state)
        deck_remaining = str(deck_totals_sim1(state))
    else:
        hand_label = format_hand_counts(state)
        deck_remaining = str(deck_totals(state))
    return TurnLog(
        turn=state.turn,
        player_hp=state.player_hp,
        player_block=state.player_block,
        energy=state.energy,
        hand_label=hand_label,
        deck_remaining=deck_remaining,
        enemy_summary=enemy_summary,
        incoming=intent,
        win_probability=decision.win_probability,
        optimal_final_hp=decision.value.final_hp if decision.value.winnable else None,
        chosen=decision.composition.label(),
        candidate_lines=[_format_dp_candidate(c, v) for c, v in decision.candidates[:8]],
    )


def _build_mc_turn_log(state: CombatState, win_p: float, decision: TurnDecision) -> TurnLog:
    enemy = state.living_enemies()[0] if state.living_enemies() else None
    intent = enemy.current_intent().label if enemy and enemy.current_intent() else "—"
    enemy_summary = (
        f"{enemy.name} HP {enemy.hp}/{enemy.max_hp} block {enemy.block}" if enemy else "—"
    )
    return TurnLog(
        turn=state.turn,
        player_hp=state.player_hp,
        player_block=state.player_block,
        energy=state.energy,
        hand_label=format_hand_counts(state),
        deck_remaining=str(deck_totals(state)),
        enemy_summary=enemy_summary,
        incoming=intent,
        win_probability=win_p,
        optimal_final_hp=None,
        chosen=decision.composition.label(),
        candidate_lines=[f"  {c.label()}: {p * 100:.1f}% (MC)" for c, p in decision.candidates[:8]],
    )


def _format_dp_candidate(comp: object, value: object) -> str:
    label = comp.label() if hasattr(comp, "label") else str(comp)
    winnable = value.winnable if hasattr(value, "winnable") else value.final_hp > -1_000_000_000
    if winnable:
        return f"  {label}: WIN at {value.final_hp} HP in {value.turns_to_end} turns"
    return f"  {label}: LOSS"


def _format_turn_log(tlog: TurnLog) -> str:
    lines = [
        f"--- Turn {tlog.turn} ---",
        f"  Player: {tlog.player_hp} HP, {tlog.player_block} block, {tlog.energy} energy",
        f"  Hand:   {tlog.hand_label}",
        f"  Deck:   {tlog.deck_remaining}",
        f"  Enemy:  {tlog.enemy_summary}",
        f"  Intent: {tlog.incoming}",
    ]
    if tlog.optimal_final_hp is not None:
        lines.append(f"  P(win): {tlog.win_probability * 100:.0f}% (exact)")
        lines.append(f"  Optimal finish: {tlog.optimal_final_hp} HP")
    else:
        lines.append(f"  P(win): {tlog.win_probability * 100:.1f}%")
    lines.append("  Candidates:")
    lines.extend(tlog.candidate_lines or ["  (none)"])
    lines.append(f"  Play:   {tlog.chosen}")
    return "\n".join(lines)


def _header(name: str, seed: int | None, solver_label: str) -> str:
    return f"=== {solver_label}: {name} (seed={seed}) ==="


def _state_snapshot(state: CombatState, *, sim: int = 0) -> str:
    enemy = state.enemies[0] if state.enemies else None
    if enemy:
        ent = (
            f"{enemy.name} HP {enemy.hp}, pattern step {enemy.pattern_index + 1}"
            f", vuln {enemy.vuln_stacks}"
        )
    else:
        ent = "no enemy"
    if sim >= 3:
        from combat_sim.deck_counts import format_hand_counts_sim3

        hand = format_hand_counts_sim3(state)
    elif sim >= 2:
        from combat_sim.deck_counts import format_hand_counts_sim2

        hand = format_hand_counts_sim2(state)
    elif sim >= 1:
        from combat_sim.deck_counts import format_hand_counts_sim1

        hand = format_hand_counts_sim1(state)
    else:
        hand = format_hand_counts(state)
    return f"Start: player {state.player_hp} HP | {ent} | hand {hand}"
