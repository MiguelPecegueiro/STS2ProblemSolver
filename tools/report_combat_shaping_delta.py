#!/usr/bin/env python3
"""Compare legacy vs new combat_turn_shaping on decisions.jsonl."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.scorer import combat_turn_shaping
from tools.recalc_run_scores import _legacy_combat_turn_shaping, _parse_combat_imm

DECISIONS_PATH = PROJECT_ROOT / "data" / "decisions.jsonl"


def main() -> int:
    legacy_by_ver: dict[str, list[float]] = defaultdict(list)
    new_by_ver: dict[str, list[float]] = defaultdict(list)
    mismatch = 0
    n = 0

    with DECISIONS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("state_type") or "").lower() not in ("monster", "elite", "boss"):
                continue
            parsed = _parse_combat_imm(row.get("immediate_reward"))
            if parsed is None:
                continue
            hp_lost, block, damage, end_reward, stored = parsed
            ver = str(row.get("agent_version") or "unknown")
            legacy = _legacy_combat_turn_shaping(hp_lost, block, damage) + end_reward
            new = combat_turn_shaping(hp_lost, block, damage) + end_reward
            legacy_by_ver[ver].append(legacy)
            new_by_ver[ver].append(new)
            n += 1
            if abs(stored - new) > 1e-4:
                mismatch += 1

    print("=== combat_score_contribution: legacy formula vs new (recomputed) ===")
    print(f"{'version':20s}  {'n':>6s}  {'legacy':>8s}  {'new':>8s}  {'delta':>8s}")
    for ver in sorted(legacy_by_ver):
        leg = legacy_by_ver[ver]
        new = new_by_ver[ver]
        avg_l = sum(leg) / len(leg)
        avg_n = sum(new) / len(new)
        print(f"{ver:20s}  {len(leg):6d}  {avg_l:8.3f}  {avg_n:8.3f}  {avg_n - avg_l:+8.3f}")

    print(f"\nStored on disk matches new formula: {n - mismatch}/{n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
