"""Exact opening expectation over C(10,5) hands for Sim 1."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from combat_sim.sim1.tuple_dp import (
    DpValue,
    FightParams,
    LOSS_HP,
    TupleStateSim1,
    make_opening_state,
    solve_tuple_sim1,
)
from combat_sim.tuple_dp import draw_orders

OPENING_TOTAL = comb(10, 5)


def opening_hand_probability(hand_s: int, hand_d: int, hand_b: int) -> float:
    if hand_s + hand_d + hand_b != 5:
        return 0.0
    if hand_s > 5 or hand_d > 4 or hand_b > 1:
        return 0.0
    return comb(5, hand_s) * comb(4, hand_d) * comb(1, hand_b) / OPENING_TOTAL


def opening_compositions_sim1() -> list[tuple[int, int, int, float]]:
    rows: list[tuple[int, int, int, float]] = []
    for hand_b in range(2):
        for hand_s in range(6):
            hand_d = 5 - hand_s - hand_b
            if hand_d < 0 or hand_d > 4:
                continue
            p = opening_hand_probability(hand_s, hand_d, hand_b)
            if p > 0:
                rows.append((hand_s, hand_d, hand_b, p))
    return rows


def remaining_deck_sim1(hand_s: int, hand_d: int, hand_b: int) -> tuple[str, ...]:
    return ("S",) * (5 - hand_s) + ("D",) * (4 - hand_d) + ("B",) * (1 - hand_b)


@dataclass(frozen=True, slots=True)
class OpeningExpectationSim1:
    win_rate: float
    expected_final_hp: float
    by_composition: list[tuple[int, int, int, float, float, float]]
    draw_orders_averaged: bool

    def format_report(self, scenario: str = "Sim 1") -> str:
        lines = [
            "=" * 60,
            f"EXACT OPENING EXPECTATION — {scenario}",
            "=" * 60,
            f"Win rate (exact):     {self.win_rate * 100:.4f}%",
            f"E[final HP]:          {self.expected_final_hp:.4f}",
            f"Draw order averaging: {'yes' if self.draw_orders_averaged else 'canonical pile'}",
            "",
            f"{'Strikes':>7} {'Defends':>7} {'Bash':>5} {'P(hand)':>10} {'P(win)':>10} {'E[HP]':>10}",
        ]
        for hs, hd, hb, prob, wr, ehp in self.by_composition:
            lines.append(
                f"{hs:>7} {hd:>7} {hb:>5} {prob * 100:>9.2f}% {wr * 100:>9.2f}% {ehp:>10.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def exact_opening_expectation_sim1(
    params: FightParams,
    *,
    shuffle_seed: int = 0,
    average_draw_orders: bool = True,
) -> OpeningExpectationSim1:
    e_hp = 0.0
    win_rate = 0.0
    rows: list[tuple[int, int, int, float, float, float]] = []

    for hand_s, hand_d, hand_b, prob in opening_compositions_sim1():
        draw_base = remaining_deck_sim1(hand_s, hand_d, hand_b)
        orders = draw_orders(draw_base) if average_draw_orders else [draw_base]

        vals: list[DpValue] = []
        for draw in orders:
            st = make_opening_state(
                params,
                hand_s=hand_s,
                hand_d=hand_d,
                hand_b=hand_b,
                draw=draw,
            )
            vals.append(
                solve_tuple_sim1(
                    st,
                    params,
                    shuffle_seed + hand_s * 100 + hand_d * 10 + hand_b + hash(draw) % 1000,
                )
            )

        avg_hp = sum(v.final_hp if v.winnable else LOSS_HP for v in vals) / len(vals)
        avg_win = sum(1.0 if v.winnable else 0.0 for v in vals) / len(vals)

        e_hp += prob * avg_hp
        win_rate += prob * avg_win
        rows.append((hand_s, hand_d, hand_b, prob, avg_win, avg_hp))

    return OpeningExpectationSim1(
        win_rate=win_rate,
        expected_final_hp=e_hp,
        by_composition=rows,
        draw_orders_averaged=average_draw_orders,
    )
