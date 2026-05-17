#!/usr/bin/env python3
"""Import STS2 .run history files into data/runs.jsonl and data/card_choices.jsonl."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.characters import normalize_character_name  # noqa: E402
from sts2_agent.scorer import run_score  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
IMPORT_DIR = DATA_DIR / "imported_runs"
# Optional: import directly from your Steam profile history folder
STEAM_HISTORY = Path(
    r"C:\Users\migue\AppData\Roaming\SlayTheSpire2\steam"
    r"\76561198011325542\profile1\saves\history"
)
RUNS_PATH = DATA_DIR / "runs.jsonl"
CARD_CHOICES_PATH = DATA_DIR / "card_choices.jsonl"
MIN_FILE_BYTES = 5 * 1024
COMBAT_TYPES = frozenset({"monster", "elite", "boss"})

def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _normalize_card_id(card: object) -> str:
    if isinstance(card, dict):
        raw = str(card.get("id") or card.get("name") or "")
    else:
        raw = str(card or "")
    if not raw:
        return ""
    if "." in raw:
        raw = raw.split(".")[-1]
    return raw.upper().replace(" ", "_")


def _normalize_character(value: object, player: dict | None = None) -> str:
    name = normalize_character_name(value)
    if name != "Unknown":
        return name
    if player:
        for card in player.get("deck") or []:
            cid = _normalize_card_id(card)
            for key in ("IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT"):
                if key in cid:
                    return normalize_character_name(key)
    return "Unknown"


def _steam_id_from_path(folder: Path) -> str | None:
    match = re.search(r"steam[/\\](\d+)[/\\]", str(folder).replace("/", "\\"))
    return match.group(1) if match else None


def _pick_player(data: dict, steam_id: str | None) -> dict:
    players = data.get("players") or []
    if not isinstance(players, list) or not players:
        return {}
    if steam_id:
        for player in players:
            if isinstance(player, dict) and str(player.get("id")) == steam_id:
                return player
    first = players[0]
    return first if isinstance(first, dict) else {}


def _pick_player_stats(floor: dict, player_id: object | None) -> dict:
    stats_list = floor.get("player_stats") or []
    if not isinstance(stats_list, list):
        return {}
    if player_id is not None:
        for stats in stats_list:
            if isinstance(stats, dict) and stats.get("player_id") == player_id:
                return stats
    for stats in stats_list:
        if isinstance(stats, dict):
            return stats
    return {}


def _is_win(data: dict) -> bool:
    if data.get("was_abandoned"):
        return False
    enc = str(data.get("killed_by_encounter") or "").upper()
    evt = str(data.get("killed_by_event") or "").upper()
    if enc == "NONE.NONE" and evt == "NONE.NONE":
        return True
    return bool(data.get("win"))


def _death_cause(data: dict) -> str | None:
    enc = str(data.get("killed_by_encounter") or "")
    evt = str(data.get("killed_by_event") or "")
    if enc and enc.upper() not in ("NONE.NONE", "NONE", ""):
        return enc
    if evt and evt.upper() not in ("NONE.NONE", "NONE", ""):
        return evt
    return None


def _iter_floors(data: dict):
    """Yield (act_index_1based, global_floor_index_1based, floor_dict)."""
    global_floor = 0
    for act_i, act in enumerate(data.get("map_point_history") or [], start=1):
        if not isinstance(act, list):
            continue
        for floor in act:
            if not isinstance(floor, dict):
                continue
            global_floor += 1
            yield act_i, global_floor, floor


def _extract_card_id_list(deck: list) -> list[str]:
    ids: list[str] = []
    for card in deck:
        cid = _normalize_card_id(card)
        if cid:
            ids.append(cid)
    return ids


def _extract_relic_id_list(relics: list) -> list[str]:
    ids: list[str] = []
    for relic in relics:
        if isinstance(relic, dict):
            raw = str(relic.get("id") or relic.get("name") or "")
        else:
            raw = str(relic)
        if not raw:
            continue
        if "." in raw:
            raw = raw.split(".")[-1]
        ids.append(raw.upper())
    return ids


def _parse_card_choices_on_floor(
    act: int,
    floor_num: int,
    floor_type: str,
    stats: dict,
    *,
    run_id: str,
    character: str,
    ascension: int,
    won: bool,
    hp_at_pick: int,
) -> tuple[dict | None, dict | None]:
    """Return (summary entry for runs.jsonl, card_choices.jsonl row)."""
    raw_choices = stats.get("card_choices") or []
    if not isinstance(raw_choices, list) or not raw_choices:
        return None, None

    offered: list[str] = []
    picked: str | None = None
    for entry in raw_choices:
        if not isinstance(entry, dict):
            continue
        card_obj = entry.get("card")
        cid = _normalize_card_id(card_obj)
        if cid:
            offered.append(cid)
        if entry.get("was_picked") and cid:
            picked = cid

    summary = {
        "floor": floor_num,
        "offered": offered,
        "picked": picked,
        "skipped": picked is None,
    }
    detail = {
        "run_id": run_id,
        "source": "human",
        "floor": floor_num,
        "act": act,
        "character": character,
        "ascension": ascension,
        "won": won,
        "offered": offered,
        "picked": picked,
        "hp_at_pick": hp_at_pick,
        "floor_type": floor_type,
    }
    return summary, detail


def parse_run_file(
    path: Path,
    *,
    steam_id: str | None = None,
) -> tuple[dict, list[dict]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict):
        return None

    player = _pick_player(data, steam_id)
    player_id = player.get("id")
    character = _normalize_character(player.get("character"), player)
    ascension = _safe_int(data.get("ascension"))
    won = _is_win(data)
    outcome = "won" if won else "lost"

    floors_reached = 0
    total_damage_taken = 0
    gold_at_death = 0
    max_hp_seen = 80
    last_hp: int | None = None

    hp_before_each_combat: list[int] = []
    hp_after_each_combat: list[int] = []
    bosses_killed = 0
    card_choices_summary: list[dict] = []
    card_choice_rows: list[dict] = []

    run_id = path.stem

    for act, floor_num, floor in _iter_floors(data):
        floors_reached = floor_num
        floor_type = str(floor.get("map_point_type") or "").lower()
        stats = _pick_player_stats(floor, player_id)

        if not stats:
            continue

        dmg = _safe_int(stats.get("damage_taken"))
        total_damage_taken += max(0, dmg)
        gold_at_death = _safe_int(stats.get("current_gold"), gold_at_death)
        max_hp_seen = max(max_hp_seen, _safe_int(stats.get("max_hp"), max_hp_seen))

        hp_now = _safe_int(stats.get("current_hp"))

        summary, detail = _parse_card_choices_on_floor(
            act,
            floor_num,
            floor_type,
            stats,
            run_id=run_id,
            character=character,
            ascension=ascension,
            won=won,
            hp_at_pick=hp_now,
        )
        if summary:
            card_choices_summary.append(summary)
        if detail:
            card_choice_rows.append(detail)

        if floor_type in COMBAT_TYPES:
            hp_after = hp_now
            hp_before = (
                last_hp
                if last_hp is not None
                else _safe_int(stats.get("max_hp"), max_hp_seen)
            )
            if hp_before <= 0:
                hp_before = max_hp_seen
            hp_before_each_combat.append(hp_before)
            hp_after_each_combat.append(hp_after)
            if floor_type == "boss" and hp_after > 0:
                bosses_killed += 1
            if hp_after > 0:
                last_hp = hp_after
        elif hp_now > 0:
            last_hp = hp_now

    act_reached = len(data.get("map_point_history") or [])

    final_deck = _extract_card_id_list(player.get("deck") or [])
    final_relics = _extract_relic_id_list(player.get("relics") or [])

    potions_at_death: list[str] = []
    for potion in player.get("potions") or []:
        if isinstance(potion, dict):
            pid = str(potion.get("id") or "")
            if "." in pid:
                pid = pid.split(".")[-1]
            if pid:
                potions_at_death.append(pid.upper())
        elif potion:
            potions_at_death.append(str(potion).upper())

    max_hp_ref = max(max_hp_seen, 1)
    hp_pcts = [
        after / max_hp_ref
        for after in hp_after_each_combat
        if max_hp_ref > 0
    ]
    avg_hp_pct = sum(hp_pcts) / len(hp_pcts) if hp_pcts else 0.0
    best_hp_pct = max(hp_pcts) if hp_pcts else 0.0
    worst_hp_pct = min(hp_pcts) if hp_pcts else 0.0

    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    run_data_for_score = {
        "floors_reached": floors_reached,
        "act_reached": act_reached,
        "avg_hp_pct_after_combat": avg_hp_pct,
        "final_deck": final_deck,
        "potions_at_death": potions_at_death,
        "bosses_killed": bosses_killed,
        "won": won,
    }

    record = {
        "run_id": run_id,
        "timestamp": mtime,
        "source": "human",
        "character": character,
        "ascension": ascension,
        "won": won,
        "floors_reached": floors_reached,
        "act_reached": act_reached,
        "cause_of_death": _death_cause(data),
        "final_deck": final_deck,
        "final_relics": final_relics,
        "total_decisions": 0,
        "total_damage_taken": total_damage_taken,
        "total_damage_dealt": 0,
        "gold_at_death": gold_at_death,
        "outcome": outcome,
        "run_score": run_score(run_data_for_score),
        "avg_hp_pct_after_combat": avg_hp_pct,
        "best_combat_hp_pct": best_hp_pct,
        "worst_combat_hp_pct": worst_hp_pct,
        "bosses_killed": bosses_killed,
        "potions_at_death": potions_at_death,
        "hp_before_each_combat": hp_before_each_combat,
        "hp_after_each_combat": hp_after_each_combat,
        "combat_rewards": [],
        "card_choices": card_choices_summary,
        "game_mode": data.get("game_mode"),
        "build_id": data.get("build_id"),
    }

    return record, card_choice_rows


def load_existing_run_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = row.get("run_id")
                if rid:
                    ids.add(str(rid))
    except OSError:
        pass
    return ids


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def import_runs(
    folder: Path,
    *,
    runs_path: Path = RUNS_PATH,
    card_choices_path: Path = CARD_CHOICES_PATH,
    min_bytes: int = MIN_FILE_BYTES,
) -> tuple[int, int, int, int, int, int]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"History folder not found: {folder}")

    steam_id = _steam_id_from_path(folder)
    existing_ids = load_existing_run_ids(runs_path)
    run_files = sorted(folder.glob("*.run"))

    imported = 0
    skipped_already = 0
    skipped_small = 0
    skipped_error = 0
    total = len(run_files)

    run_rows: list[dict] = []
    choice_rows: list[dict] = []

    for path in run_files:
        run_id = path.stem
        if run_id in existing_ids:
            skipped_already += 1
            continue
        try:
            size = path.stat().st_size
        except OSError:
            skipped_error += 1
            continue
        if size < min_bytes:
            skipped_small += 1
            continue

        parsed = parse_run_file(path, steam_id=steam_id)
        if not parsed:
            skipped_error += 1
            continue

        record, choices = parsed
        run_rows.append(record)
        choice_rows.extend(choices)
        existing_ids.add(run_id)
        imported += 1

    append_jsonl(runs_path, run_rows)
    append_jsonl(card_choices_path, choice_rows)

    skipped = skipped_already + skipped_small + skipped_error
    return imported, skipped, total, skipped_already, skipped_small, skipped_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import STS2 .run files into training JSONL datasets",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=IMPORT_DIR,
        help="Folder of .run files to import (default: data/imported_runs)",
    )
    parser.add_argument(
        "--runs-out",
        type=Path,
        default=RUNS_PATH,
        help="Output runs.jsonl path (default: data/runs.jsonl)",
    )
    parser.add_argument(
        "--choices-out",
        type=Path,
        default=CARD_CHOICES_PATH,
        help="Output card_choices.jsonl path (default: data/card_choices.jsonl)",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=MIN_FILE_BYTES,
        help="Skip .run files smaller than this (default: 5120)",
    )
    args = parser.parse_args()

    import_dir = args.folder.expanduser().resolve()
    import_dir.mkdir(parents=True, exist_ok=True)

    try:
        imported, skipped, total, skip_dup, skip_small, skip_err = import_runs(
            import_dir,
            runs_path=args.runs_out,
            card_choices_path=args.choices_out,
            min_bytes=args.min_bytes,
        )
    except FileNotFoundError as exc:
        print(exc)
        return 1

    print(f"Scanning: {import_dir}")
    if total == 0:
        print("  No .run files found - drop files into data/imported_runs/ and run again")
    print(f"Imported {imported}/{total} runs ({skipped} skipped)")
    if skip_dup:
        print(f"  already in {args.runs_out.name}: {skip_dup}")
    if skip_small:
        print(f"  too small (<{args.min_bytes} bytes): {skip_small}")
    if skip_err:
        print(f"  parse/read errors: {skip_err}")
    if imported == 0 and skip_dup == total:
        print("  (nothing new - delete rows from runs.jsonl or use a fresh file to re-import)")
    print(f"  runs -> {args.runs_out}")
    print(f"  card choices -> {args.choices_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
