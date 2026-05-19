"""CLI: optimal DP fight log or batch win-rate report.

Examples:
  py -m combat_sim --scenario jaw_worm
  py -m combat_sim --sim 1 --scenario jaw_worm_sim1
  py -m combat_sim --sim 1 --opening-expectation --scenario jaw_worm_sim1 --fast
"""

from __future__ import annotations

import argparse

from combat_sim.math_solver import MathSolverConfig
from combat_sim.runner import run_batch, run_logged_fight
from combat_sim.scenarios import (
    bygone_effigy_sim0,
    bygone_effigy_sim1,
    bygone_effigy_sim2,
    bygone_effigy_sim3,
    skulking_colony_sim3,
    jaw_worm,
    jaw_worm_sim1,
    jaw_worm_sim2,
    jaw_worm_sim3,
    slime_boss_minion,
)
from combat_sim.tuple_dp import FightParams, exact_opening_expectation

SCENARIOS = {
    "jaw_worm": ("Jaw Worm (Sim 0)", jaw_worm),
    "jaw_worm_sim1": ("Jaw Worm (Sim 1)", jaw_worm_sim1),
    "jaw_worm_sim2": ("Jaw Worm (Sim 2)", jaw_worm_sim2),
    "jaw_worm_sim3": ("Jaw Worm (Sim 3)", jaw_worm_sim3),
    "bygone_effigy_sim0": ("Bygone Effigy (Sim 0 deck)", bygone_effigy_sim0),
    "bygone_effigy_sim1": ("Bygone Effigy (Sim 1 deck)", bygone_effigy_sim1),
    "bygone_effigy_sim2": ("Bygone Effigy (Sim 2 deck)", bygone_effigy_sim2),
    "bygone_effigy_sim3": ("Bygone Effigy (Sim 3 deck)", bygone_effigy_sim3),
    "skulking_colony_sim3": ("Skulking Colony (Sim 3 deck)", skulking_colony_sim3),
    "slime": ("Slime (Sim 0)", slime_boss_minion),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Combat sim — optimal DP (default) or MC verify")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="jaw_worm")
    parser.add_argument("--sim", type=int, choices=[0, 1, 2, 3], default=None, help="Force sim version")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=0, metavar="N")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="MC: low rollouts; opening: skip draw-order averaging",
    )
    parser.add_argument("--mc", action="store_true", help="Use Monte Carlo solver (Sim 0 only)")
    parser.add_argument(
        "--greedy-batch",
        action="store_true",
        help="Batch only: fast greedy solver",
    )
    parser.add_argument("--decision-rollouts", type=int, default=40)
    parser.add_argument("--estimate-rollouts", type=int, default=120)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--opening-expectation",
        action="store_true",
        help="Exact E[HP] and win rate over all opening hands",
    )
    args = parser.parse_args()

    if args.sim is not None:
        sim = args.sim
    elif args.scenario == "bygone_effigy_sim0":
        sim = 0
    elif args.scenario == "bygone_effigy_sim1":
        sim = 1
    elif args.scenario == "bygone_effigy_sim2":
        sim = 2
    elif args.scenario == "bygone_effigy_sim3":
        sim = 3
    elif args.scenario == "skulking_colony_sim3":
        sim = 3
    elif "sim3" in args.scenario:
        sim = 3
    elif "sim2" in args.scenario:
        sim = 2
    elif "sim1" in args.scenario:
        sim = 1
    else:
        sim = 0
    label, scenario_fn = SCENARIOS[args.scenario]
    solver = "mc" if args.mc else "optimal"
    config = MathSolverConfig(
        decision_rollouts=args.decision_rollouts,
        estimate_rollouts=args.estimate_rollouts,
        max_turns=args.max_turns,
        seed=args.seed,
    )

    if args.opening_expectation:
        if sim >= 3:
            from combat_sim.sim3.opening import exact_opening_expectation_sim3
            from combat_sim.sim3.tuple_dp import FightParams as FP3

            params = FP3(80, 40, (("A", 7), ("B", 5), ("A", 11)))
            report = exact_opening_expectation_sim3(
                params,
                shuffle_seed=args.seed,
                average_draw_orders=not args.fast,
            )
        elif sim >= 2:
            from combat_sim.sim2.opening import exact_opening_expectation_sim2
            from combat_sim.sim2.tuple_dp import FightParams as FP2

            params = FP2(80, 40, (("A", 7), ("B", 5), ("A", 11)))
            report = exact_opening_expectation_sim2(
                params,
                shuffle_seed=args.seed,
                average_draw_orders=not args.fast,
            )
        elif sim >= 1:
            from combat_sim.sim1.opening import exact_opening_expectation_sim1
            from combat_sim.sim1.tuple_dp import FightParams as FP1

            params = FP1(80, 40, (("A", 7), ("B", 5), ("A", 11)))
            report = exact_opening_expectation_sim1(
                params,
                shuffle_seed=args.seed,
                average_draw_orders=not args.fast,
            )
        else:
            if args.scenario == "jaw_worm":
                params = FightParams(80, 40, (("A", 7), ("B", 5), ("A", 11)))
            else:
                params = FightParams(50, 12, (("A", 8),))
            report = exact_opening_expectation(
                params,
                shuffle_seed=args.seed,
                average_draw_orders=not args.fast,
            )
        print(report.format_report(label))
        return

    if args.batch > 0:
        batch_solver = "greedy" if args.greedy_batch else solver
        if sim >= 1 and args.mc:
            print("Monte Carlo is Sim 0 only; using optimal DP.")
            batch_solver = "optimal"
        report = run_batch(
            scenario_fn,
            scenario_name=label,
            runs=args.batch,
            solver=batch_solver,
            config=config,
            fast=args.fast,
            base_seed=args.base_seed,
            max_turns=args.max_turns,
            sim=sim,
        )
        print(report.format_report())
        return

    if sim >= 1 and args.mc:
        print("Monte Carlo is Sim 0 only; using optimal DP.")
    run_logged_fight(
        scenario_fn,
        scenario_name=label,
        seed=args.seed,
        solver="optimal" if sim >= 1 else solver,
        config=config,
        max_turns=args.max_turns,
        verbose=True,
        sim=sim,
    )


if __name__ == "__main__":
    main()
