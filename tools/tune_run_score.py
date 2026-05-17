#!/usr/bin/env python3
"""Grid-search run_score weights against human runs in data/runs.jsonl."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.scorer import run_score as current_run_score  # noqa: E402

RUNS_PATH = PROJECT_ROOT / "data" / "runs.jsonl"

FLOOR_MULTS = (5, 8, 10, 15)
HP_MULTS = (25, 50, 75, 100)
WIN_BONUSES = (300, 500, 750, 1000)
DECK_SCORE_OPTS = (True, False)
POTION_PENALTIES = (0, -5, -8, -15)
BOSS_BONUSES = (50, 75, 100)

ACT_BONUS_PER_ACT = 60  # fixed (not in search grid)


@dataclass(frozen=True)
class FormulaParams:
    floor_mult: float
    hp_mult: float
    win_bonus: float
    use_deck_score: bool
    potion_penalty: float
    boss_bonus: float

    def label(self) -> str:
        deck = "deck_on" if self.use_deck_score else "deck_off"
        return (
            f"floor*{self.floor_mult:g} hp*{self.hp_mult:g} "
            f"win={self.win_bonus:g} {deck} "
            f"potion*{self.potion_penalty:g} boss*{self.boss_bonus:g}"
        )


@dataclass
class FormulaStats:
    params: FormulaParams
    win_mean: float
    loss_mean: float
    win_n: int
    loss_n: int

    @property
    def gap(self) -> float:
        return self.win_mean - self.loss_mean

    @property
    def ratio(self) -> float:
        if self.loss_mean <= 0:
            return float("inf") if self.win_mean > 0 else 0.0
        return self.win_mean / self.loss_mean


def _run_won(run: dict) -> bool:
    if run.get("won") is not None:
        return bool(run.get("won"))
    outcome = str(run.get("outcome") or "").lower()
    return outcome in ("win", "won", "victory")


def _deck_score(deck_size: int, enabled: bool) -> float:
    if not enabled:
        return 0.0
    if deck_size <= 12:
        return 20.0
    if deck_size <= 16:
        return 10.0
    if deck_size <= 20:
        return 0.0
    return -(deck_size - 20) * 2.0


def score_run(run: dict, p: FormulaParams) -> float:
    floors = int(run.get("floors_reached") or 0)
    act = int(run.get("act_reached") or 1)
    avg_hp_remaining = float(run.get("avg_hp_pct_after_combat") or 0.0)
    deck = run.get("final_deck") or []
    deck_size = len(deck) if isinstance(deck, list) else 0
    potions = run.get("potions_at_death") or []
    potion_count = len(potions) if isinstance(potions, list) else 0
    bosses_killed = int(run.get("bosses_killed") or 0)
    won = _run_won(run)

    total = (
        floors * p.floor_mult
        + (act - 1) * ACT_BONUS_PER_ACT
        + avg_hp_remaining * p.hp_mult
        + _deck_score(deck_size, p.use_deck_score)
        + potion_count * p.potion_penalty
        + (p.win_bonus if won else 0.0)
        + bosses_killed * p.boss_bonus
    )
    return max(0.0, float(total))


def load_human_runs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    runs: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("source") or "").lower() == "human":
                runs.append(row)
    return runs


def evaluate(params: FormulaParams, runs: list[dict]) -> FormulaStats:
    win_scores: list[float] = []
    loss_scores: list[float] = []
    for run in runs:
        s = score_run(run, params)
        if _run_won(run):
            win_scores.append(s)
        else:
            loss_scores.append(s)
    win_mean = sum(win_scores) / len(win_scores) if win_scores else 0.0
    loss_mean = sum(loss_scores) / len(loss_scores) if loss_scores else 0.0
    return FormulaStats(
        params=params,
        win_mean=win_mean,
        loss_mean=loss_mean,
        win_n=len(win_scores),
        loss_n=len(loss_scores),
    )


def evaluate_baseline(runs: list[dict]) -> FormulaStats:
    win_scores: list[float] = []
    loss_scores: list[float] = []
    for run in runs:
        s = current_run_score(run)
        if _run_won(run):
            win_scores.append(s)
        else:
            loss_scores.append(s)
    win_mean = sum(win_scores) / len(win_scores) if win_scores else 0.0
    loss_mean = sum(loss_scores) / len(loss_scores) if loss_scores else 0.0
    baseline_params = FormulaParams(
        floor_mult=15,
        hp_mult=100,
        win_bonus=1000,
        use_deck_score=False,
        potion_penalty=0,
        boss_bonus=100,
    )
    return FormulaStats(
        params=baseline_params,
        win_mean=win_mean,
        loss_mean=loss_mean,
        win_n=len(win_scores),
        loss_n=len(loss_scores),
    )


def print_stats(stats: FormulaStats, *, title: str | None = None, rank: int | None = None) -> None:
    prefix = f"{rank:>2}. " if rank is not None else "    "
    heading = title or stats.params.label()
    print(f"{prefix}{heading}")
    if title:
        print(f"      {stats.params.label()}")
    print(
        f"      win_mean={stats.win_mean:7.1f}  loss_mean={stats.loss_mean:7.1f}  "
        f"gap={stats.gap:7.1f}  ratio={stats.ratio:5.2f}  "
        f"(n win={stats.win_n} loss={stats.loss_n})"
    )


def main() -> int:
    runs = load_human_runs(RUNS_PATH)
    if not runs:
        print(f"No human runs found in {RUNS_PATH}")
        return 1

    print(f"Loaded {len(runs)} human runs from {RUNS_PATH}")
    baseline = evaluate_baseline(runs)
    print_stats(baseline, title="BASELINE (current sts2_agent.scorer.run_score)")
    print()

    results: list[FormulaStats] = []
    for floor_mult, hp_mult, win_bonus, use_deck, potion_pen, boss_bonus in product(
        FLOOR_MULTS,
        HP_MULTS,
        WIN_BONUSES,
        DECK_SCORE_OPTS,
        POTION_PENALTIES,
        BOSS_BONUSES,
    ):
        params = FormulaParams(
            floor_mult=float(floor_mult),
            hp_mult=float(hp_mult),
            win_bonus=float(win_bonus),
            use_deck_score=use_deck,
            potion_penalty=float(potion_pen),
            boss_bonus=float(boss_bonus),
        )
        results.append(evaluate(params, runs))

    results.sort(key=lambda s: (s.gap, s.ratio), reverse=True)

    print(f"Tested {len(results)} formula variants")
    print("Top 10 by gap (then win/loss ratio):\n")
    for i, stats in enumerate(results[:10], start=1):
        print_stats(stats, rank=i)

    best = results[0]
    print()
    print(
        f"Best vs baseline: gap {best.gap - baseline.gap:+.1f}, "
        f"ratio {best.ratio:.2f} vs {baseline.ratio:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
