"""Exact opening expectation over C(11,5) hands for Sim 2."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from combat_sim.sim2.tuple_dp import (
    DpValue,
    FightParams,
    LOSS_HP,
    make_opening_state,
    solve_tuple_sim2,
)
from combat_sim.tuple_dp import draw_orders

OPENING_TOTAL = comb(11, 5)


def opening_hand_probability(
    hand_s: int,
    hand_d: int,
    hand_b: int,
    hand_bl: int,
) -> float:
    if hand_s + hand_d + hand_b + hand_bl != 5:
        return 0.0
    if hand_s > 5 or hand_d > 4 or hand_b > 1 or hand_bl > 1:
        return 0.0
    return (
        comb(5, hand_s)
        * comb(4, hand_d)
        * comb(1, hand_b)
        * comb(1, hand_bl)
        / OPENING_TOTAL
    )


def opening_compositions_sim2() -> list[tuple[int, int, int, int, float]]:
    rows: list[tuple[int, int, int, int, float]] = []
    for hand_bl in range(2):
        for hand_b in range(2):
            for hand_s in range(6):
                hand_d = 5 - hand_s - hand_b - hand_bl
                if hand_d < 0 or hand_d > 4:
                    continue
                p = opening_hand_probability(hand_s, hand_d, hand_b, hand_bl)
                if p > 0:
                    rows.append((hand_s, hand_d, hand_b, hand_bl, p))
    return rows


def remaining_deck_sim2(
    hand_s: int,
    hand_d: int,
    hand_b: int,
    hand_bl: int,
) -> tuple[str, ...]:
    return (
        ("S",) * (5 - hand_s)
        + ("D",) * (4 - hand_d)
        + ("B",) * (1 - hand_b)
        + ("L",) * (1 - hand_bl)
    )


@dataclass(frozen=True, slots=True)
class OpeningExpectationSim2:
    win_rate: float
    expected_final_hp: float
    by_composition: list[tuple[int, int, int, int, float, float, float]]
    draw_orders_averaged: bool

    def format_report(self, scenario: str = "Sim 2") -> str:
        lines = [
            "=" * 60,
            f"EXACT OPENING EXPECTATION — {scenario}",
            "=" * 60,
            f"Win rate (exact):     {self.win_rate * 100:.4f}%",
            f"E[final HP]:          {self.expected_final_hp:.4f}",
            f"Draw order averaging: {'yes' if self.draw_orders_averaged else 'canonical pile'}",
            "",
            f"{'St':>3} {'Def':>3} {'Bsh':>3} {'BL':>3} {'P(hand)':>10} {'P(win)':>10} {'E[HP]':>10}",
        ]
        for hs, hd, hb, hbl, prob, wr, ehp in self.by_composition:
            lines.append(
                f"{hs:>3} {hd:>3} {hb:>3} {hbl:>3} {prob * 100:>9.2f}% {wr * 100:>9.2f}% {ehp:>10.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


def exact_opening_expectation_sim2(
    params: FightParams,
    *,
    shuffle_seed: int = 0,
    average_draw_orders: bool = True,
) -> OpeningExpectationSim2:
    e_hp = 0.0
    win_rate = 0.0
    rows: list[tuple[int, int, int, int, float, float, float]] = []

    for hand_s, hand_d, hand_b, hand_bl, prob in opening_compositions_sim2():
        draw_base = remaining_deck_sim2(hand_s, hand_d, hand_b, hand_bl)
        orders = draw_orders(draw_base) if average_draw_orders else [draw_base]

        vals: list[DpValue] = []
        for draw in orders:
            st = make_opening_state(
                params,
                hand_s=hand_s,
                hand_d=hand_d,
                hand_b=hand_b,
                hand_bl=hand_bl,
                draw=draw,
            )
            vals.append(
                solve_tuple_sim2(
                    st,
                    params,
                    shuffle_seed
                    + hand_s * 1000
                    + hand_d * 100
                    + hand_b * 10
                    + hand_bl
                    + hash(draw) % 1000,
                )
            )

        avg_hp = sum(v.final_hp if v.winnable else LOSS_HP for v in vals) / len(vals)
        avg_win = sum(1.0 if v.winnable else 0.0 for v in vals) / len(vals)

        e_hp += prob * avg_hp
        win_rate += prob * avg_win
        rows.append((hand_s, hand_d, hand_b, hand_bl, prob, avg_win, avg_hp))

    return OpeningExpectationSim2(
        win_rate=win_rate,
        expected_final_hp=e_hp,
        by_composition=rows,
        draw_orders_averaged=average_draw_orders,
    )
