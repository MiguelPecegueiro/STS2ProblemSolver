"""Import enemy patterns into cache/enemy_patterns.json.

Preferred source: reference/enemies.csv (editable table).
Legacy source: reference/Slay the Spire 2 Reference.xlsx

Usage:
    py tools/import_enemy_patterns.py
    py tools/import_enemy_patterns.py --csv reference/enemies.csv
    py tools/import_enemy_patterns.py --xlsx reference/Slay the Spire 2 Reference.xlsx
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "reference" / "enemies.csv"
DEFAULT_XLSX = PROJECT_ROOT / "reference" / "Slay the Spire 2 Reference.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "cache" / "enemy_patterns.json"

MOVE_COLS = [f"Move {i}" for i in range(1, 7)]
SHEETS = ("Monsters", "Elites", "Bosses")

DAMAGE_RE = re.compile(
    r"damage\s+(\d+)(?:\s*\(\s*\d+\s*\))?(?:\s*x\s*(\d+))?",
    re.IGNORECASE,
)
BLOCK_RE = re.compile(r"block\s+(\d+)", re.IGNORECASE)
CYCLE_ARROW_RE = re.compile(
    r"cycles?\s+through\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)
ALT_RE = re.compile(
    r"alternates?\s+between\s+(.+?)\s+and\s+(.+?)(?:\.|$)",
    re.IGNORECASE,
)
THEN_EVERY_RE = re.compile(
    r"starts?\s+with\s+([^.,]+).*?(?:then\s+)?uses?\s+([^.,]+)\s+every\s+turn",
    re.IGNORECASE,
)


def _parse_damage(text: str) -> int:
    total = 0
    for match in DAMAGE_RE.finditer(text):
        base = int(match.group(1))
        hits = int(match.group(2) or 1)
        total += base * hits
    return total


def _parse_block(text: str) -> int:
    match = BLOCK_RE.search(text)
    return int(match.group(1)) if match else 0


def _parse_pattern(pattern: str, move_names: list[str]) -> dict:
    """Best-effort parse of Attack Pattern column into a cycle hint."""
    if not pattern or not move_names:
        return {"kind": "unknown", "cycle": move_names}

    text = str(pattern).strip()
    lower = text.lower()

    arrow = CYCLE_ARROW_RE.search(text)
    if arrow:
        parts = [p.strip() for p in re.split(r"\s*->\s*", arrow.group(1)) if p.strip()]
        cycle = [_fuzzy_move_name(p, move_names) for p in parts]
        cycle = [c for c in cycle if c]
        if cycle:
            return {"kind": "cycle", "cycle": cycle, "raw": text}

    alt = ALT_RE.search(text)
    if alt:
        a = _fuzzy_move_name(alt.group(1), move_names)
        b = _fuzzy_move_name(alt.group(2), move_names)
        if a and b:
            return {"kind": "alternate", "cycle": [a, b], "raw": text}

    then = THEN_EVERY_RE.search(text)
    if then:
        first = _fuzzy_move_name(then.group(1), move_names)
        repeat = _fuzzy_move_name(then.group(2), move_names)
        if repeat:
            return {
                "kind": "then_repeat",
                "opening": [first] if first else [],
                "cycle": [repeat],
                "raw": text,
            }

    if "every turn" in lower and move_names:
        return {"kind": "repeat_last", "cycle": [move_names[-1]], "raw": text}

    return {"kind": "unknown", "cycle": move_names, "raw": text}


def _fuzzy_move_name(fragment: str, move_names: list[str]) -> str | None:
    frag = fragment.strip().lower()
    if not frag:
        return None
    for name in move_names:
        nl = name.lower()
        if frag in nl or nl in frag:
            return name
    for name in move_names:
        if frag.split()[0] in name.lower():
            return name
    return None


def _collect_moves(rows: list[pd.Series]) -> list[dict]:
    moves: list[dict] = []
    for col in MOVE_COLS:
        parts: list[str] = []
        for row in rows:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        if not parts:
            continue
        name = parts[0]
        body = " ".join(parts)
        lower = body.lower()
        moves.append(
            {
                "name": name,
                "text": body,
                "damage": _parse_damage(body),
                "block": _parse_block(body),
                "is_attack": _parse_damage(body) > 0,
                "is_buff": any(k in lower for k in ("strength", "ritual", "artifact", "dexterity")),
                "is_debuff": any(k in lower for k in ("weak", "frail", "vulnerable", "dazed", "wound")),
            }
        )
    return moves


def _parse_sheet(df: pd.DataFrame, category: str) -> list[dict]:
    name_col = df.columns[0]
    pattern_col = "Attack Pattern" if "Attack Pattern" in df.columns else None
    hp_col = "HP" if "HP" in df.columns else None
    monsters: list[dict] = []
    i = 0
    while i < len(df):
        raw_name = df.iloc[i].get(name_col)
        if pd.isna(raw_name) or not str(raw_name).strip():
            i += 1
            continue
        name = str(raw_name).strip()
        if name.startswith("(") and "minion" in name.lower():
            i += 1
            continue

        block_rows = [df.iloc[i]]
        j = i + 1
        while j < len(df):
            next_name = df.iloc[j].get(name_col)
            if pd.notna(next_name) and str(next_name).strip():
                break
            block_rows.append(df.iloc[j])
            j += 1

        moves = _collect_moves(block_rows)
        pattern_text = ""
        if pattern_col:
            val = block_rows[0].get(pattern_col)
            if pd.notna(val):
                pattern_text = str(val).strip()
        hp_text = ""
        if hp_col:
            val = block_rows[0].get(hp_col)
            if pd.notna(val):
                hp_text = str(val).strip()

        move_names = [m["name"] for m in moves]
        monsters.append(
            {
                "name": name,
                "category": category,
                "hp": hp_text,
                "moves": moves,
                "pattern": _parse_pattern(pattern_text, move_names),
                "pattern_text": pattern_text,
                "notes": str(block_rows[0].get("Notes") or "").strip()
                if "Notes" in block_rows[0]
                else "",
            }
        )
        i = j
    return monsters


def import_patterns_from_csv(csv_path: Path) -> dict:
    """Load one row per move from reference/enemies.csv."""
    groups: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("enemy_name") or "").strip()
            if not name:
                continue
            entry = groups.setdefault(
                name,
                {
                    "name": name,
                    "category": "monster",
                    "hp": "",
                    "moves": [],
                    "pattern": {"kind": "unknown", "cycle": []},
                    "pattern_text": "",
                    "notes": "",
                },
            )
            if (row.get("category") or "").strip():
                entry["category"] = row["category"].strip()
            if (row.get("notes") or "").strip():
                entry["notes"] = row["notes"].strip()
            kind = (row.get("pattern_kind") or "").strip()
            cycle_raw = (row.get("pattern_cycle") or "").strip()
            if kind:
                cycle = [c.strip() for c in cycle_raw.split("|") if c.strip()]
                entry["pattern"] = {"kind": kind, "cycle": cycle, "raw": cycle_raw}
            move_name = (row.get("move_name") or "").strip()
            if not move_name:
                continue
            aliases_raw = (row.get("api_aliases") or "").strip()
            aliases = [a.strip() for a in aliases_raw.split("|") if a.strip()]
            damage = int(float(row.get("damage") or 0))
            block = int(float(row.get("block") or 0))
            verified_run = (row.get("verified_run_id") or "").strip()
            entry["moves"].append(
                {
                    "name": move_name,
                    "text": move_name,
                    "damage": damage,
                    "block": block,
                    "is_attack": damage > 0,
                    "is_buff": False,
                    "is_debuff": False,
                    "aliases": aliases,
                    "verified_run_id": verified_run,
                }
            )

    monsters = []
    by_name: dict[str, dict] = {}
    for entry in groups.values():
        if not entry["moves"]:
            entry["pattern"] = {"kind": "unknown", "cycle": []}
        elif not entry["pattern"].get("cycle"):
            entry["pattern"]["cycle"] = [m["name"] for m in entry["moves"]]
        monsters.append(entry)
        by_name[entry["name"].lower()] = entry

    return {
        "source": str(csv_path.name),
        "monsters": monsters,
        "by_name": by_name,
    }


def import_patterns(xlsx_path: Path) -> dict:
    all_monsters: list[dict] = []
    by_name: dict[str, dict] = {}

    for sheet in SHEETS:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        category = sheet.lower().rstrip("s")
        if category == "monsters":
            category = "monster"
        entries = _parse_sheet(df, category)
        all_monsters.extend(entries)
        for entry in entries:
            key = entry["name"].lower().strip()
            by_name[key] = entry

    return {
        "source": str(xlsx_path.name),
        "monsters": all_monsters,
        "by_name": by_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import STS2 enemy patterns")
    parser.add_argument("--csv", type=Path, default=None, help="CSV table (default if file exists)")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    csv_path = args.csv or DEFAULT_CSV
    if csv_path.exists():
        data = import_patterns_from_csv(csv_path)
        source = csv_path
    elif args.xlsx.exists():
        data = import_patterns(args.xlsx)
        source = args.xlsx
    else:
        raise SystemExit(
            f"No reference found. Add {DEFAULT_CSV} or {DEFAULT_XLSX}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Imported {len(data['monsters'])} enemies from {source.name} -> {args.out}")


if __name__ == "__main__":
    main()
