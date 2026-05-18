#!/usr/bin/env python3
"""Report run_score distribution by agent version (before stored vs after formula)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.scorer import damage_efficiency_penalty, deck_quality_score, run_score  # noqa: E402

RUNS_PATH = PROJECT_ROOT / "data" / "runs.jsonl"


def _stats(scores: list[float]) -> dict[str, float | int] | None:
    if not scores:
        return None
    return {
        "n": len(scores),
        "avg": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


def _run_data(row: dict) -> dict:
    won = row.get("won")
    if won is None:
        won = str(row.get("outcome") or "").lower() in ("win", "won", "victory")
    return {
        "floors_reached": row.get("floors_reached"),
        "act_reached": row.get("act_reached"),
        "avg_hp_pct_after_combat": row.get("avg_hp_pct_after_combat"),
        "bosses_killed": row.get("bosses_killed"),
        "won": bool(won),
        "combat_summary": row.get("combat_summary"),
    }


def _has_summary(combat_summary: object) -> bool:
    return isinstance(combat_summary, list) and len(combat_summary) > 0


def main() -> int:
    rows = [
        json.loads(line)
        for line in RUNS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    by_ver_before: dict[str, list[float]] = defaultdict(list)
    by_ver_after: dict[str, list[float]] = defaultdict(list)
    by_pen: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        if row.get("source") == "human":
            continue
        ver = str(row.get("agent_version") or "unknown")
        rd = _run_data(row)
        before = float(row.get("run_score") or 0.0)
        after = run_score(rd)
        by_ver_before[ver].append(before)
        by_ver_after[ver].append(after)
        if _has_summary(rd.get("combat_summary")):
            by_pen[ver].append(damage_efficiency_penalty(rd.get("combat_summary")))

    print("=== AGENT RUNS: avg / min / max by version ===")
    print("BEFORE (stored run_score on disk):")
    for ver in sorted(by_ver_before):
        s = _stats(by_ver_before[ver])
        assert s is not None
        print(f"  {ver:20s} n={s['n']:4d}  avg={s['avg']:7.1f}  min={s['min']:7.0f}  max={s['max']:7.0f}")

    print()
    print("AFTER (hp*50 + deck quality + damage penalty):")
    for ver in sorted(by_ver_after):
        s = _stats(by_ver_after[ver])
        assert s is not None
        print(f"  {ver:20s} n={s['n']:4d}  avg={s['avg']:7.1f}  min={s['min']:7.0f}  max={s['max']:7.0f}")

    print()
    print("=== AVG DECK SCORE (runs with final_deck) ===")
    by_deck: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("source") == "human":
            continue
        deck = row.get("final_deck")
        if not isinstance(deck, list) or not deck:
            continue
        ver = str(row.get("agent_version") or "unknown")
        by_deck[ver].append(deck_quality_score(deck))
    for ver in sorted(by_deck):
        vals = by_deck[ver]
        print(f"  {ver:20s} n={len(vals):4d}  avg_deck={sum(vals) / len(vals):6.1f}")

    print()
    print("=== AVG PENALTY (runs with combat_summary only) ===")
    for ver in sorted(by_pen):
        vals = by_pen[ver]
        print(
            f"  {ver:20s} n={len(vals):4d}  avg_penalty={sum(vals) / len(vals):6.1f}  "
            f"min={min(vals):6.0f}  max={max(vals):6.0f}"
        )

    print()
    print("=== VERSION ORDERING (avg score) ===")

    def _order(d: dict[str, list[float]]) -> list[tuple[str, dict]]:
        items = [(v, _stats(s)) for v, s in d.items()]
        items = [(v, s) for v, s in items if s is not None]
        return sorted(items, key=lambda x: x[1]["avg"])

    before_order = " < ".join(f"{v}({s['avg']:.0f})" for v, s in _order(by_ver_before))
    after_order = " < ".join(f"{v}({s['avg']:.0f})" for v, s in _order(by_ver_after))
    print(f"Before: {before_order}")
    print(f"After:  {after_order}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
