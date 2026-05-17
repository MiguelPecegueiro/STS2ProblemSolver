#!/usr/bin/env python3
"""Merge per-instance training shards into data/decisions.jsonl and data/runs.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCES_DIR = PROJECT_ROOT / "data" / "instances"
DECISIONS_PATH = PROJECT_ROOT / "data" / "decisions.jsonl"
RUNS_PATH = PROJECT_ROOT / "data" / "runs.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _append_jsonl(path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return len(rows)


def _decision_key(row: dict) -> tuple:
    action = row.get("action_taken")
    action_blob = json.dumps(action, sort_keys=True, default=str) if action is not None else ""
    return (
        str(row.get("run_id") or ""),
        str(row.get("timestamp") or ""),
        action_blob,
    )


def _run_key(row: dict) -> str:
    return str(row.get("run_id") or "")


def merge_shards(
    instances_dir: Path,
    *,
    decisions_path: Path = DECISIONS_PATH,
    runs_path: Path = RUNS_PATH,
) -> tuple[int, int]:
    """Append shard JSONL into main training files; dedupe by run_id / decision key."""
    shard_dirs = sorted(
        p for p in instances_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )

    existing_runs = _read_jsonl(runs_path)
    seen_run_ids = {_run_key(r) for r in existing_runs if _run_key(r)}

    existing_decisions = _read_jsonl(decisions_path)
    seen_decision_keys = {_decision_key(r) for r in existing_decisions}

    new_runs: list[dict] = []
    new_decisions: list[dict] = []

    for shard_dir in shard_dirs:
        shard_runs = _read_jsonl(shard_dir / "runs.jsonl")
        for row in shard_runs:
            rid = _run_key(row)
            if not rid or rid in seen_run_ids:
                continue
            seen_run_ids.add(rid)
            new_runs.append(row)

        shard_decisions = _read_jsonl(shard_dir / "decisions.jsonl")
        for row in shard_decisions:
            key = _decision_key(row)
            if key in seen_decision_keys:
                continue
            seen_decision_keys.add(key)
            new_decisions.append(row)

    runs_added = _append_jsonl(runs_path, new_runs)
    decisions_added = _append_jsonl(decisions_path, new_decisions)
    return runs_added, decisions_added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=DEFAULT_INSTANCES_DIR,
        help="Root directory containing per-instance shard folders",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DECISIONS_PATH,
        help="Target decisions.jsonl (appends)",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=RUNS_PATH,
        help="Target runs.jsonl (appends)",
    )
    args = parser.parse_args()

    instances_dir = args.instances_dir
    if not instances_dir.is_absolute():
        instances_dir = PROJECT_ROOT / instances_dir

    if not instances_dir.exists():
        print(f"No instances directory: {instances_dir}", file=sys.stderr)
        return 1

    runs_added, decisions_added = merge_shards(
        instances_dir,
        decisions_path=args.decisions,
        runs_path=args.runs,
    )
    print(
        f"Merged from {instances_dir}: +{runs_added} runs, +{decisions_added} decisions "
        f"-> {args.runs.name}, {args.decisions.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
