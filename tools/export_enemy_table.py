"""Export cache/enemy_patterns.json to reference/enemies.csv for editing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "cache" / "enemy_patterns.json"
DEFAULT_OUT = PROJECT_ROOT / "reference" / "enemies.csv"

COLUMNS = (
    "enemy_name",
    "category",
    "pattern_kind",
    "pattern_cycle",
    "move_name",
    "damage",
    "block",
    "api_aliases",
    "notes",
    "verified_run_id",
)


def export_table(data: dict, out_path: Path) -> int:
    rows_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for entry in data.get("monsters") or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or ""
            category = entry.get("category") or "monster"
            pattern = entry.get("pattern") or {}
            kind = pattern.get("kind") or "unknown"
            cycle = pattern.get("cycle") or [m.get("name") for m in entry.get("moves") or []]
            cycle_str = "|".join(c for c in cycle if c)
            notes = str(entry.get("notes") or "").replace("\n", " ")
            moves = entry.get("moves") or []
            if not moves:
                writer.writerow(
                    {
                        "enemy_name": name,
                        "category": category,
                        "pattern_kind": kind,
                        "pattern_cycle": cycle_str,
                        "move_name": "",
                        "damage": 0,
                        "block": 0,
                        "api_aliases": "",
                        "notes": notes,
                        "verified_run_id": "",
                    }
                )
                rows_written += 1
                continue
            for i, move in enumerate(moves):
                aliases = move.get("aliases") or []
                if isinstance(aliases, str):
                    aliases = [a.strip() for a in aliases.split("|") if a.strip()]
                writer.writerow(
                    {
                        "enemy_name": name,
                        "category": category,
                        "pattern_kind": kind if i == 0 else "",
                        "pattern_cycle": cycle_str if i == 0 else "",
                        "move_name": move.get("name") or "",
                        "damage": int(move.get("damage") or 0),
                        "block": int(move.get("block") or 0),
                        "api_aliases": "|".join(aliases),
                        "notes": notes if i == 0 else "",
                        "verified_run_id": str(move.get("verified_run_id") or ""),
                    }
                )
                rows_written += 1
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export enemy patterns to CSV table")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.cache.exists():
        raise SystemExit(f"Cache not found: {args.cache} (run import_enemy_patterns.py first)")
    data = json.loads(args.cache.read_text(encoding="utf-8"))
    n = export_table(data, args.out)
    print(f"Wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
