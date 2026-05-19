"""Exact opening expectation over C(12,5) hands for Sim 3."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from combat_sim.sim3.tuple_dp import (
    DpValue,
    FightParams,
    LOSS_HP,
    make_opening_state,
    solve_tuple_sim3,
)
from combat_sim.tuple_dp import draw_orders

OPENING_TOTAL = comb(12, 5)


def opening_hand_probability(
    hand_s: int,
    hand_d: int,
    hand_b: int,
    hand_bl: int,
    hand_inf: int,
) -> float:
    if hand_s + hand_d + hand_b + hand_bl + hand_inf != 5:
        return 0.0
    if hand_s > 5 or hand_d > 4 or hand_b > 1 or hand_bl > 1 or hand_inf > 1:
        return 0.0
    return (
        comb(5, hand_s)
        * comb(4, hand_d)
        * comb(1, hand_b)
        * comb(1, hand_bl)
        * comb(1, hand_inf)
        / OPENING_TOTAL
    )


def opening_compositions_sim3() -> list[tuple[int, int, int, int, int, float]]:
    rows: list[tuple[int, int, int, int, int, float]] = []
    for hand_inf in range(2):
        for hand_bl in range(2):
            for hand_b in range(2):
                for hand_s in range(6):
                    hand_d = 5 - hand_s - hand_b - hand_bl - hand_inf
                    if hand_d < 0 or hand_d > 4:
                        continue
                    p = opening_hand_probability(hand_s, hand_d, hand_b, hand_bl, hand_inf)
                    if p > 0:
                        rows.append((hand_s, hand_d, hand_b, hand_bl, hand_inf, p))
    return rows


def remaining_deck_sim3(
    hand_s: int,
    hand_d: int,
    hand_b: int,
    hand_bl: int,
    hand_inf: int,
) -> tuple[str, ...]:
    return (
        ("S",) * (5 - hand_s)
        + ("D",) * (4 - hand_d)
        + ("B",) * (1 - hand_b)
        + ("L",) * (1 - hand_bl)
        + ("I",) * (1 - hand_inf)
    )


@dataclass(frozen=True, slots=True)
class OpeningExpectationSim3:
    win_rate: float
    expected_final_hp: float
    by_composition: list[tuple[int, int, int, int, int, float, float, float]]
    draw_orders_averaged: bool

    def format_report(self, scenario: str = "Sim 3") -> str:
        lines = [
            "=" * 60,
            f"EXACT OPENING EXPECTATION — {scenario}",
            "=" * 60,
            f"Win rate (exact):     {self.win_rate * 100:.4f}%",
            f"E[final HP]:          {self.expected_final_hp:.4f}",
            f"Draw order averaging: {'yes' if self.draw_orders_averaged else 'canonical pile'}",
            "",
            f"{'St':>3} {'Def':>3} {'Bsh':>3} {'BL':>3} {'INF':>3} {'P(hand)':>10} {'P(win)':>10} {'E[HP]':>10}",
        ]
        for hs, hd, hb, hbl, hinf, prob, wr, ehp in self.by_composition:
            lines.append(
                f"{hs:>3} {hd:>3} {hb:>3} {hbl:>3} {hinf:>3} {prob * 100:>9.2f}% {wr * 100:>9.2f}% {ehp:>10.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def exact_opening_expectation_sim3(
    params: FightParams,
    *,
    shuffle_seed: int = 0,
    average_draw_orders: bool = True,
) -> OpeningExpectationSim3:
    e_hp = 0.0
    win_rate = 0.0
    rows: list[tuple[int, int, int, int, int, float, float, float]] = []

    for hand_s, hand_d, hand_b, hand_bl, hand_inf, prob in opening_compositions_sim3():
        draw_base = remaining_deck_sim3(hand_s, hand_d, hand_b, hand_bl, hand_inf)
        orders = draw_orders(draw_base) if average_draw_orders else [draw_base]

        vals: list[DpValue] = []
        for draw in orders:
            st = make_opening_state(
                params,
                hand_s=hand_s,
                hand_d=hand_d,
                hand_b=hand_b,
                hand_bl=hand_bl,
                hand_inf=hand_inf,
                draw=draw,
            )
            vals.append(
                solve_tuple_sim3(
                    st,
                    params,
                    shuffle_seed
                    + hand_s * 10000
                    + hand_d * 1000
                    + hand_b * 100
                    + hand_bl * 10
                    + hand_inf
                    + hash(draw) % 1000,
                )
            )

        avg_hp = sum(v.final_hp if v.winnable else LOSS_HP for v in vals) / len(vals)
        avg_win = sum(1.0 if v.winnable else 0.0 for v in vals) / len(vals)

        e_hp += prob * avg_hp
        win_rate += prob * avg_win
        rows.append((hand_s, hand_d, hand_b, hand_bl, hand_inf, prob, avg_win, avg_hp))

    return OpeningExpectationSim3(
        win_rate=win_rate,
        expected_final_hp=e_hp,
        by_composition=rows,
        draw_orders_averaged=average_draw_orders,
    )
