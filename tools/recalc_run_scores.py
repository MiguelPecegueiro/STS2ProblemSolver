#!/usr/bin/env python3
"""Recalculate run_score on data/runs.jsonl (and decisions.jsonl) after formula changes.

Each run row already stores floors_reached, act_reached, avg_hp_pct_after_combat,
bosses_killed, and won - no need to re-import .run files unless those fields are missing.

Usage:
  py tools/recalc_run_scores.py              # update runs + decisions
  py tools/recalc_run_scores.py --dry-run    # print stats only
  py tools/recalc_run_scores.py --runs-only  # skip decisions.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.scorer import run_score  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RUNS_PATH = DATA_DIR / "runs.jsonl"
DECISIONS_PATH = DATA_DIR / "decisions.jsonl"


def _run_won(row: dict) -> bool:
    if row.get("won") is not None:
        return bool(row.get("won"))
    outcome = str(row.get("outcome") or "").lower()
    return outcome in ("win", "won", "victory")


def _run_data_from_row(row: dict) -> dict:
    return {
        "floors_reached": row.get("floors_reached"),
        "act_reached": row.get("act_reached"),
        "avg_hp_pct_after_combat": row.get("avg_hp_pct_after_combat"),
        "bosses_killed": row.get("bosses_killed"),
        "won": _run_won(row),
        "combat_summary": row.get("combat_summary"),
        "final_deck": row.get("final_deck"),
    }


def recalc_runs(path: Path, *, dry_run: bool) -> dict[str, float]:
    """Rewrite runs.jsonl; return run_id -> new score."""
    if not path.exists():
        raise FileNotFoundError(path)

    updated: list[dict] = []
    scores: dict[str, float] = {}
    changed = 0
    missing_hp = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = str(row.get("run_id") or "")
            if row.get("avg_hp_pct_after_combat") is None:
                missing_hp += 1
            old = float(row.get("run_score") or 0.0)
            new = run_score(_run_data_from_row(row))
            if abs(old - new) > 0.01:
                changed += 1
            row["run_score"] = new
            if rid:
                scores[rid] = new
            updated.append(row)

    if not dry_run:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as out:
            for row in updated:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(path)
        print(f"Wrote {path} ({len(updated)} runs, backup at {backup.name})")
    else:
        print(f"[dry-run] Would update {path} ({len(updated)} runs)")

    print(
        f"  runs: {len(updated)} total, {changed} scores changed, "
        f"{missing_hp} missing avg_hp_pct (scored as 0)"
    )
    return scores


def recalc_decisions(path: Path, scores: dict[str, float], *, dry_run: bool) -> None:
    if not path.exists():
        print(f"No {path.name} - skipped")
        return

    updated_rows: list[str] = []
    touched = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            row = json.loads(raw)
            rid = str(row.get("run_id") or "")
            new_score = scores.get(rid)
            if new_score is None:
                updated_rows.append(raw)
                continue
            outcome = row.get("run_outcome")
            if isinstance(outcome, dict) and (
                outcome.get("run_score") is not None or outcome.get("reward") is not None
            ):
                old = float(outcome.get("run_score") or outcome.get("reward") or 0.0)
                if abs(old - new_score) > 0.01:
                    touched += 1
                outcome["run_score"] = new_score
                outcome["reward"] = new_score
                row["run_outcome"] = outcome
                updated_rows.append(json.dumps(row, ensure_ascii=False))
            else:
                updated_rows.append(raw)

    if not dry_run:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as out:
            for raw in updated_rows:
                out.write(raw + "\n")
        tmp.replace(path)
        print(f"Wrote {path} (backup at {backup.name})")
    else:
        print(f"[dry-run] Would update {path}")

    print(f"  decisions: {touched} run_outcome scores updated")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without writing files",
    )
    parser.add_argument(
        "--runs-only",
        action="store_true",
        help="Only update runs.jsonl (training still prefers decision run_outcome)",
    )
    args = parser.parse_args()

    scores = recalc_runs(RUNS_PATH, dry_run=args.dry_run)
    if not args.runs_only:
        recalc_decisions(DECISIONS_PATH, scores, dry_run=args.dry_run)

    if scores:
        vals = sorted(scores.values())
        print(
            f"  new score range: {vals[0]:.0f} .. {vals[-1]:.0f} "
            f"(median {vals[len(vals) // 2]:.0f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
