#!/usr/bin/env python3
"""Report Phase B filter counts and combat_score_contribution stats."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset import (  # noqa: E402
    DEFAULT_DECISIONS_PATH,
    DEFAULT_RUNS_PATH,
    load_decision_rows,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    print("=== Phase B training filter (clean_only=True) ===")
    _rows, _scores, meta = load_decision_rows(
        DEFAULT_DECISIONS_PATH,
        runs_path=DEFAULT_RUNS_PATH,
        min_run_score_percentile=25.0,
        clean_only=True,
    )
    print(f"  decisions kept: {meta['rows_kept']}")
    print(f"  human kept: {meta['kept_human']}")
    print(f"  agent Phase B kept: {meta['kept_agent_phase_b']}")
    print(f"  agent discarded (no combat_summary): {meta['rows_discarded_phase_b']}")
    print(f"  Phase B runs in runs.jsonl: {meta['phase_b_run_count']}")

    print()
    print("=== Phase B training filter (clean_only=False, baseline) ===")
    rows_all, _scores2, meta_off = load_decision_rows(
        DEFAULT_DECISIONS_PATH,
        runs_path=DEFAULT_RUNS_PATH,
        min_run_score_percentile=25.0,
        clean_only=False,
    )
    print(f"  decisions kept: {meta_off['rows_kept']} (delta {meta_off['rows_kept'] - meta['rows_kept']:+d})")

    print()
    print("=== Combat shaping (legacy vs current formula) ===")
    from tools.report_combat_shaping_delta import main as shaping_main

    return shaping_main()


if __name__ == "__main__":
    raise SystemExit(main())
