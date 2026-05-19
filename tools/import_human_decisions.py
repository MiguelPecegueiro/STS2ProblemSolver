#!/usr/bin/env python3
"""Convert human card_choices.jsonl rows into synthetic decisions.jsonl for BC training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CARD_CHOICES = PROJECT_ROOT / "data" / "card_choices.jsonl"
DEFAULT_DECISIONS = PROJECT_ROOT / "data" / "decisions.jsonl"


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _normalize_offered(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip().upper())
        elif isinstance(item, dict):
            cid = str(item.get("id") or item.get("name") or "").strip()
            if cid:
                if "." in cid:
                    cid = cid.split(".")[-1]
                out.append(cid.upper().replace(" ", "_"))
    return out


def _normalize_picked(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in ("none", "null"):
        return None
    if "." in text:
        text = text.split(".")[-1]
    return text.upper().replace(" ", "_")


def _build_snapshot(row: dict, offered: list[str]) -> dict[str, Any]:
    """Minimal snapshot: offered cards as pseudo-hand for feature encoding."""
    hp = _safe_int(row.get("hp_at_pick"), 70)
    max_hp = max(hp, 1)
    hand = [
        {
            "id": cid,
            "name": cid.replace("_", " "),
            "cost": 1,
            "type": "unknown",
            "can_play": True,
        }
        for cid in offered[:10]
    ]
    return {
        "player_hp": hp,
        "player_max_hp": max_hp,
        "player_block": 0,
        "player_energy": 0,
        "hand": hand,
        "draw_pile_count": 0,
        "discard_pile_count": 0,
        "relics": [],
        "potions": [],
        "status_effects": [],
        "enemies": [],
        "attack_ratio_in_draw": 0.0,
        "block_ratio_in_draw": 0.0,
        "high_value_cards_in_draw": 0.0,
        "expected_block_next_turn": 0,
        "expected_damage_next_turn": 0,
        "card_reward_offered": offered,
    }


def card_choice_to_decision(row: dict) -> dict[str, Any] | None:
    if str(row.get("source") or "").lower() != "human":
        return None

    run_id = str(row.get("run_id") or "").strip()
    if not run_id:
        return None

    offered = _normalize_offered(row.get("offered"))
    if not offered:
        return None

    floor = _safe_int(row.get("floor"))
    act = _safe_int(row.get("act"), 1)
    picked = _normalize_picked(row.get("picked"))

    if picked is not None:
        try:
            card_index = offered.index(picked)
        except ValueError:
            return None
        action_taken = {"action": "select_card_reward", "card_index": card_index}
        action_reasoning = f"synthetic human card_reward: picked {picked} [{card_index}]"
        card_reward_picked = picked
    else:
        action_taken = {"action": "skip_card_reward"}
        action_reasoning = "synthetic human card_reward: skip"
        card_index = None
        card_reward_picked = None

    decision: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": None,
        "source": "human",
        "agent_version": "human",
        "game_version": row.get("game_version") or "unknown",
        "floor": floor,
        "act": act,
        "state_type": "card_reward",
        "state_snapshot": _build_snapshot(row, offered),
        "action_taken": action_taken,
        "action_reasoning": action_reasoning,
        "immediate_reward": 0,
        "run_outcome": None,
        "card_reward_offered": offered,
    }
    if card_reward_picked:
        decision["card_reward_picked"] = card_reward_picked
    if card_index is not None:
        decision["card_index"] = card_index
    return decision


def _decision_dedupe_key(row: dict) -> tuple[str, int] | None:
    run_id = str(row.get("run_id") or "").strip()
    if not run_id:
        return None
    return run_id, _safe_int(row.get("floor"))


def load_existing_keys(decisions_path: Path) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    if not decisions_path.exists():
        return keys
    with decisions_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = _decision_dedupe_key(row)
            if key is not None:
                keys.add(key)
    return keys


def import_human_decisions(
    *,
    card_choices_path: Path,
    decisions_path: Path,
) -> dict[str, int]:
    """Append synthetic card_reward decisions; return stats."""
    stats = {
        "card_choices_read": 0,
        "skipped_not_human": 0,
        "skipped_no_run_id": 0,
        "skipped_no_offered": 0,
        "skipped_unmapped_pick": 0,
        "skipped_duplicate": 0,
        "added_select": 0,
        "added_skip": 0,
    }

    if not card_choices_path.exists():
        raise FileNotFoundError(f"Card choices file not found: {card_choices_path}")

    seen = load_existing_keys(decisions_path)
    new_rows: list[dict[str, Any]] = []

    with card_choices_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["card_choices_read"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if str(row.get("source") or "").lower() != "human":
                stats["skipped_not_human"] += 1
                continue
            if not str(row.get("run_id") or "").strip():
                stats["skipped_no_run_id"] += 1
                continue

            offered = _normalize_offered(row.get("offered"))
            if not offered:
                stats["skipped_no_offered"] += 1
                continue

            floor = _safe_int(row.get("floor"))
            run_id = str(row.get("run_id") or "").strip()
            dedupe = (run_id, floor)
            if dedupe in seen:
                stats["skipped_duplicate"] += 1
                continue

            decision = card_choice_to_decision(row)
            if decision is None:
                stats["skipped_unmapped_pick"] += 1
                continue

            seen.add(dedupe)
            new_rows.append(decision)
            if decision["action_taken"]["action"] == "skip_card_reward":
                stats["added_skip"] += 1
            else:
                stats["added_select"] += 1

    if new_rows:
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with decisions_path.open("a", encoding="utf-8") as fh:
            for row in new_rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    stats["added_total"] = stats["added_select"] + stats["added_skip"]
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card-choices",
        type=Path,
        default=DEFAULT_CARD_CHOICES,
        help="Input card_choices.jsonl (default: data/card_choices.jsonl)",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_DECISIONS,
        help="Output decisions.jsonl to append (default: data/decisions.jsonl)",
    )
    args = parser.parse_args()

    try:
        stats = import_human_decisions(
            card_choices_path=args.card_choices,
            decisions_path=args.decisions,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Read {stats['card_choices_read']} card_choices rows")
    print(f"Added {stats['added_total']} synthetic decisions to {args.decisions}")
    print(f"  select_card_reward: {stats['added_select']}")
    print(f"  skip_card_reward: {stats['added_skip']}")
    if stats["skipped_duplicate"]:
        print(f"  skipped (duplicate run_id+floor): {stats['skipped_duplicate']}")
    if stats["skipped_not_human"]:
        print(f"  skipped (not human): {stats['skipped_not_human']}")
    if stats["skipped_no_offered"]:
        print(f"  skipped (no offered): {stats['skipped_no_offered']}")
    if stats["skipped_unmapped_pick"]:
        print(f"  skipped (picked not in offered): {stats['skipped_unmapped_pick']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
