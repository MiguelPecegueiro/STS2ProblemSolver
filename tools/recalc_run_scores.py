#!/usr/bin/env python3
"""Recalculate run_score and combat immediate_reward after formula changes.

- runs.jsonl: run_score from floors, HP, deck quality, etc.
- decisions.jsonl: combat_score_contribution from stored hp/block/damage + combat_end_reward

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
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.scorer import combat_turn_shaping, run_score  # noqa: E402

COMBAT_STATE_TYPES = frozenset({"monster", "elite", "boss"})

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


def _legacy_combat_turn_shaping(
    hp_lost_this_turn: int,
    block_applied: int,
    damage_dealt: int,
) -> float:
    return (
        -float(hp_lost_this_turn) * 2.0
        + float(block_applied) * 0.15
        + float(damage_dealt) * 0.25
    )


def _parse_combat_imm(imm: object) -> tuple[int, int, int, float, float] | None:
    if not isinstance(imm, dict):
        return None
    if not any(k in imm for k in ("hp_lost_this_turn", "block_applied", "damage_dealt")):
        return None
    try:
        hp_lost = int(imm.get("hp_lost_this_turn") or 0)
        block = int(imm.get("block_applied") or 0)
        damage = int(imm.get("damage_dealt") or 0)
        end_reward = float(imm.get("combat_end_reward") or 0.0)
        stored = float(imm.get("combat_score_contribution") or 0.0)
    except (TypeError, ValueError):
        return None
    return hp_lost, block, damage, end_reward, stored


def recalc_combat_shaping(path: Path, *, dry_run: bool) -> dict[str, tuple[float, float, int]]:
    """Rewrite combat immediate_reward dicts; return version -> (avg_before, avg_after, n)."""
    if not path.exists():
        print(f"No {path.name} - skipped combat shaping")
        return {}

    legacy_by_ver: dict[str, list[float]] = defaultdict(list)
    after_by_ver: dict[str, list[float]] = defaultdict(list)
    updated_rows: list[str] = []
    touched = 0
    skipped = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            row = json.loads(raw)
            st = str(row.get("state_type") or "").lower()
            if st not in COMBAT_STATE_TYPES:
                updated_rows.append(raw)
                continue

            imm = row.get("immediate_reward")
            parsed = _parse_combat_imm(imm)
            if parsed is None:
                skipped += 1
                updated_rows.append(raw)
                continue

            hp_lost, block, damage, end_reward, stored_contrib = parsed
            ver = str(row.get("agent_version") or "unknown")
            legacy_contrib = _legacy_combat_turn_shaping(hp_lost, block, damage) + end_reward
            legacy_by_ver[ver].append(legacy_contrib)

            shaping = combat_turn_shaping(hp_lost, block, damage)
            new_contrib = shaping + end_reward

            if abs(stored_contrib - new_contrib) > 1e-6:
                touched += 1

            updated = dict(imm)
            updated["combat_score_contribution"] = new_contrib
            row["immediate_reward"] = updated
            after_by_ver[ver].append(new_contrib)
            updated_rows.append(json.dumps(row, ensure_ascii=False))

    summary: dict[str, tuple[float, float, int]] = {}
    for ver in sorted(set(legacy_by_ver) | set(after_by_ver)):
        b = legacy_by_ver[ver]
        a = after_by_ver[ver]
        if not b:
            continue
        summary[ver] = (sum(b) / len(b), sum(a) / len(a), len(b))

    if not dry_run:
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as out:
            for raw in updated_rows:
                out.write(raw + "\n")
        tmp.replace(path)
        print(f"Wrote {path} combat shaping (backup at {backup.name})")
    else:
        print(f"[dry-run] Would update {path} combat shaping")

    print(
        f"  combat decisions: {touched} contributions changed, "
        f"{skipped} rows skipped (no combat dict)"
    )

    print()
    print("=== AVG combat_score_contribution (combat decisions) ===")
    print(f"{'version':20s}  {'n':>6s}  {'legacy':>8s}  {'new':>8s}  {'delta':>8s}")
    for ver, (avg_b, avg_a, n) in sorted(summary.items()):
        print(f"{ver:20s}  {n:6d}  {avg_b:8.3f}  {avg_a:8.3f}  {avg_a - avg_b:+8.3f}")

    return summary


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
        recalc_combat_shaping(DECISIONS_PATH, dry_run=args.dry_run)
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
