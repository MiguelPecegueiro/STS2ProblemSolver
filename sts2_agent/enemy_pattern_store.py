"""Enemy pattern compendium: CSV source of truth, audit meta, agent proposals."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
REFERENCE_DIR = PROJECT_ROOT / "reference"
CSV_PATH = REFERENCE_DIR / "enemies.csv"
META_PATH = REFERENCE_DIR / "enemy_pattern_meta.json"
PROPOSALS_PATH = PROJECT_ROOT / "data" / "enemy_pattern_proposals.jsonl"
IMPORT_SCRIPT = PROJECT_ROOT / "tools" / "import_enemy_patterns.py"

CSV_COLUMNS = (
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

SOURCE_MANUAL = "manual"
SOURCE_AGENT = "agent"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def field_key(enemy_name: str, move_name: str | None, field: str) -> str:
    move = move_name or "__enemy__"
    return f"{enemy_name}::{move}::{field}"


def _ensure_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[list(CSV_COLUMNS)]


def load_table() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=list(CSV_COLUMNS))
    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    return _ensure_csv_columns(df)


def load_meta() -> dict[str, Any]:
    if not META_PATH.exists():
        return {"version": 1, "fields": {}}
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def save_meta(meta: dict[str, Any]) -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def record_field_change(
    meta: dict[str, Any],
    *,
    enemy_name: str,
    move_name: str | None,
    field: str,
    value: str,
    source: str,
    run_id: str | None = None,
    verified_run_id: str | None = None,
) -> None:
    key = field_key(enemy_name, move_name, field)
    prev = meta.setdefault("fields", {}).get(key, {})
    entry = {
        "value": value,
        "updated_at": _utc_now(),
        "source": source,
        "run_id": run_id or "",
        "previous_value": prev.get("value", ""),
    }
    if verified_run_id:
        entry["verified_run_id"] = verified_run_id
        entry["verified_at"] = _utc_now()
    elif prev.get("verified_run_id"):
        entry["verified_run_id"] = prev.get("verified_run_id")
        entry["verified_at"] = prev.get("verified_at", "")
    meta["fields"][key] = entry


def save_table(
    df: pd.DataFrame,
    *,
    source: str = SOURCE_MANUAL,
    run_id: str | None = None,
    changed: list[tuple[str, str | None, str, str]] | None = None,
) -> None:
    """Persist CSV, update meta for changed fields, rebuild agent cache."""
    df = _ensure_csv_columns(df.copy())
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    if changed:
        for enemy_name, move_name, field, value in changed:
            record_field_change(
                meta,
                enemy_name=enemy_name,
                move_name=move_name,
                field=field,
                value=value,
                source=source,
                run_id=run_id,
            )
    save_meta(meta)
    df.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    rebuild_agent_cache()


def rebuild_agent_cache() -> None:
    if not IMPORT_SCRIPT.exists():
        return
    subprocess.run(
        [sys.executable, str(IMPORT_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
    )
    # Invalidate in-process KB singleton
    try:
        import sts2_agent.enemy_patterns as ep

        ep._kb = None  # noqa: SLF001
    except ImportError:
        pass


def meta_as_rows(meta: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key, entry in (meta.get("fields") or {}).items():
        parts = key.split("::", 2)
        if len(parts) != 3:
            continue
        enemy, move, field = parts
        rows.append(
            {
                "enemy_name": enemy,
                "move_name": "" if move == "__enemy__" else move,
                "field": field,
                "value": str(entry.get("value", "")),
                "previous_value": str(entry.get("previous_value", "")),
                "updated_at": str(entry.get("updated_at", "")),
                "source": str(entry.get("source", "")),
                "run_id": str(entry.get("run_id", "")),
                "verified_run_id": str(entry.get("verified_run_id", "")),
                "verified_at": str(entry.get("verified_at", "")),
            }
        )
    rows.sort(key=lambda r: r.get("updated_at", ""), reverse=True)
    return rows


def get_field_meta(enemy_name: str, move_name: str | None, field: str) -> dict[str, str]:
    meta = load_meta()
    return (meta.get("fields") or {}).get(field_key(enemy_name, move_name, field), {})


def _read_proposals() -> list[dict[str, Any]]:
    if not PROPOSALS_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    with PROPOSALS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _write_proposals(rows: list[dict[str, Any]]) -> None:
    PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROPOSALS_PATH.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def propose_agent_update(
    *,
    enemy_name: str,
    move_name: str | None,
    field: str,
    proposed_value: str,
    old_value: str = "",
    run_id: str | None = None,
    note: str = "",
) -> str | None:
    """Queue an agent-suggested change. Returns proposal id or None if duplicate pending."""
    key = field_key(enemy_name, move_name, field)
    rows = _read_proposals()
    for row in rows:
        if row.get("status") == "pending" and row.get("field_key") == key:
            if str(row.get("proposed_value")) == str(proposed_value):
                return None
    proposal_id = str(uuid.uuid4())[:8]
    rows.append(
        {
            "id": proposal_id,
            "timestamp": _utc_now(),
            "status": "pending",
            "field_key": key,
            "enemy_name": enemy_name,
            "move_name": move_name or "",
            "field": field,
            "old_value": old_value,
            "proposed_value": proposed_value,
            "run_id": run_id or "",
            "note": note,
        }
    )
    _write_proposals(rows)
    return proposal_id


def list_proposals(*, status: str | None = "pending") -> list[dict[str, Any]]:
    rows = _read_proposals()
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return sorted(rows, key=lambda r: r.get("timestamp", ""), reverse=True)


def _apply_proposal_to_df(df: pd.DataFrame, proposal: dict[str, Any]) -> pd.DataFrame:
    enemy = proposal["enemy_name"]
    move = (proposal.get("move_name") or "").strip()
    field = proposal["field"]
    value = str(proposal.get("proposed_value", ""))

    if field in ("pattern_kind", "pattern_cycle", "category", "notes") and not move:
        mask = df["enemy_name"] == enemy
        if mask.any():
            idx = df[mask].index[0]
            df.at[idx, field] = value
        return df

    mask = (df["enemy_name"] == enemy) & (df["move_name"] == move)
    if not mask.any():
        return df
    idx = df[mask].index[0]
    df.at[idx, field] = value
    return df


def accept_proposal(proposal_id: str) -> bool:
    rows = _read_proposals()
    target = next((r for r in rows if r.get("id") == proposal_id), None)
    if not target or target.get("status") != "pending":
        return False

    df = load_table()
    df = _apply_proposal_to_df(df, target)
    move_name = (target.get("move_name") or "").strip() or None
    save_table(
        df,
        source=SOURCE_AGENT,
        run_id=target.get("run_id") or None,
        changed=[
            (
                target["enemy_name"],
                move_name,
                target["field"],
                str(target.get("proposed_value", "")),
            )
        ],
    )

    for row in rows:
        if row.get("id") == proposal_id:
            row["status"] = "accepted"
            row["resolved_at"] = _utc_now()
    _write_proposals(rows)
    return True


def reject_proposal(proposal_id: str) -> bool:
    rows = _read_proposals()
    found = False
    for row in rows:
        if row.get("id") == proposal_id:
            row["status"] = "rejected"
            row["resolved_at"] = _utc_now()
            found = True
    if found:
        _write_proposals(rows)
    return found


def mark_verified_in_run(
    enemy_name: str,
    move_name: str,
    run_id: str,
    *,
    field: str = "move_match",
) -> None:
    """Mark a move pattern as verified by an agent/human run observation."""
    if not enemy_name or not move_name or not run_id:
        return
    df = load_table()
    mask = (df["enemy_name"] == enemy_name) & (df["move_name"] == move_name)
    if not mask.any():
        return
    idx = df[mask].index[0]
    existing = str(df.at[idx, "verified_run_id"] or "").strip()
    if existing == run_id:
        return
    df.at[idx, "verified_run_id"] = run_id
    meta = load_meta()
    record_field_change(
        meta,
        enemy_name=enemy_name,
        move_name=move_name,
        field=field,
        value=move_name,
        source=SOURCE_AGENT,
        run_id=run_id,
        verified_run_id=run_id,
    )
    save_meta(meta)
    df.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    rebuild_agent_cache()


def enemies_grouped(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    groups: dict[str, pd.DataFrame] = {}
    for name, grp in df.groupby("enemy_name", sort=True):
        groups[str(name)] = grp.reset_index(drop=True)
    return groups


def delete_enemy(df: pd.DataFrame, enemy_name: str) -> pd.DataFrame:
    return df[df["enemy_name"] != enemy_name].reset_index(drop=True)


def add_enemy_row(
    df: pd.DataFrame,
    *,
    enemy_name: str,
    category: str = "monster",
    pattern_kind: str = "unknown",
    pattern_cycle: str = "",
    move_name: str = "",
    damage: int = 0,
    block: int = 0,
    api_aliases: str = "",
    notes: str = "",
) -> pd.DataFrame:
    row = {col: "" for col in CSV_COLUMNS}
    row.update(
        {
            "enemy_name": enemy_name,
            "category": category,
            "pattern_kind": pattern_kind,
            "pattern_cycle": pattern_cycle,
            "move_name": move_name,
            "damage": str(damage),
            "block": str(block),
            "api_aliases": api_aliases,
            "notes": notes,
            "verified_run_id": "",
        }
    )
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
