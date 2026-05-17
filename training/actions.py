"""Action vocabulary: map action dicts to class ids and back."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

MAX_CARD_INDEX = 9
MAX_GENERIC_INDEX = 9
MAX_POTION_SLOT = 2
MAX_TARGET_SLOT = 4

# Actions that use "index" parameter (map, rewards, events, etc.)
MAX_ENEMIES = 5


def _enemy_target_slot(enemies: list[dict], target_id: str | None) -> int | None:
    if not target_id:
        return None
    tid = str(target_id)
    for idx, enemy in enumerate(enemies[:MAX_ENEMIES]):
        if not isinstance(enemy, dict):
            continue
        eid = str(enemy.get("entity_id") or enemy.get("id") or "")
        if eid and eid == tid:
            return idx
    return None


INDEX_ACTIONS = frozenset(
    {
        "choose_map_node",
        "claim_reward",
        "select_card",
        "choose_event_option",
        "choose_rest_option",
        "claim_treasure_relic",
        "shop_purchase",
    }
)


def action_to_key(action: dict | None, *, snapshot: dict | None = None) -> str | None:
    """Canonical string key for an action dict."""
    if not action or not isinstance(action, dict):
        return None
    name = str(action.get("action") or "").strip()
    if not name:
        return None

    if name == "play_card":
        idx = int(action.get("card_index", 0))
        idx = max(0, min(idx, MAX_CARD_INDEX))
        key = f"play_card:{idx}"
        target = action.get("target")
        if target is not None and snapshot:
            enemies = [
                e for e in (snapshot.get("enemies") or []) if isinstance(e, dict)
            ]
            slot = _enemy_target_slot(enemies, str(target))
            if slot is not None:
                slot = max(0, min(slot, MAX_TARGET_SLOT))
                key += f":tgt:{slot}"
        return key

    if name in INDEX_ACTIONS:
        idx = int(action.get("index", 0))
        idx = max(0, min(idx, MAX_GENERIC_INDEX))
        return f"{name}:{idx}"

    if name in ("select_card_reward", "combat_select_card"):
        idx = int(action.get("card_index", 0))
        idx = max(0, min(idx, MAX_CARD_INDEX))
        return f"{name}:{idx}"

    if name == "use_potion":
        slot = int(action.get("slot", 0))
        slot = max(0, min(slot, MAX_POTION_SLOT))
        key = f"use_potion:{slot}"
        if action.get("target") is not None and snapshot:
            enemies = [
                e for e in (snapshot.get("enemies") or []) if isinstance(e, dict)
            ]
            tslot = _enemy_target_slot(enemies, str(action.get("target")))
            if tslot is not None:
                tslot = max(0, min(tslot, MAX_TARGET_SLOT))
                key += f":tgt:{tslot}"
        return key

    if name == "discard_potion":
        slot = int(action.get("slot", 0))
        slot = max(0, min(slot, MAX_POTION_SLOT))
        return f"discard_potion:{slot}"

    if name in (
        "end_turn",
        "proceed",
        "confirm_selection",
        "combat_confirm_selection",
        "advance_dialogue",
    ):
        return name

    return name


def build_action_vocab(decisions_path: Path) -> dict[str, int]:
    """Scan decisions.jsonl and assign stable class ids (sorted keys)."""
    keys: set[str] = set()
    with decisions_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = action_to_key(
                row.get("action_taken"),
                snapshot=row.get("state_snapshot"),
            )
            if key:
                keys.add(key)
    ordered = sorted(keys)
    return {key: idx for idx, key in enumerate(ordered)}


def encode_action(
    action: dict | None,
    vocab: dict[str, int],
    *,
    snapshot: dict | None = None,
    unknown_index: int | None = None,
) -> int | None:
    key = action_to_key(action, snapshot=snapshot)
    if key is None:
        return None
    if key in vocab:
        return vocab[key]
    return unknown_index


def decode_action(class_id: int, id_to_key: dict[int, str]) -> dict[str, Any]:
    """Reconstruct a best-effort action dict from class id."""
    key = id_to_key.get(int(class_id), "")
    if not key:
        return {"action": "end_turn"}

    if key == "end_turn":
        return {"action": "end_turn"}
    if key == "proceed":
        return {"action": "proceed"}
    if key == "confirm_selection":
        return {"action": "confirm_selection"}
    if key == "combat_confirm_selection":
        return {"action": "combat_confirm_selection"}
    if key == "advance_dialogue":
        return {"action": "advance_dialogue"}

    if key.startswith("play_card:"):
        m = re.match(r"play_card:(\d+)(?::tgt:(\d+))?", key)
        if not m:
            return {"action": "end_turn"}
        out: dict[str, Any] = {"action": "play_card", "card_index": int(m.group(1))}
        if m.group(2) is not None:
            out["target"] = f"ENEMY_{m.group(2)}"
        return out

    if key.startswith("use_potion:"):
        m = re.match(r"use_potion:(\d+)(?::tgt:(\d+))?", key)
        if not m:
            return {"action": "end_turn"}
        out = {"action": "use_potion", "slot": int(m.group(1))}
        if m.group(2) is not None:
            out["target"] = f"ENEMY_{m.group(2)}"
        return out

    if key.startswith("discard_potion:"):
        m = re.match(r"discard_potion:(\d+)", key)
        if m:
            return {"action": "discard_potion", "slot": int(m.group(1))}

    if ":" in key:
        action_name, idx_str = key.split(":", 1)
        idx_str = idx_str.split(":tgt:")[0]
        if action_name in ("select_card_reward", "combat_select_card"):
            return {"action": action_name, "card_index": int(idx_str)}
        return {"action": action_name, "index": int(idx_str)}

    return {"action": key}


def vocab_metadata(vocab: dict[str, int]) -> dict[str, Any]:
    id_to_key = {idx: key for key, idx in vocab.items()}
    return {
        "num_actions": len(vocab),
        "action_to_id": vocab,
        "id_to_action_key": {str(k): v for k, v in id_to_key.items()},
        "max_card_index": MAX_CARD_INDEX,
        "max_generic_index": MAX_GENERIC_INDEX,
        "max_potion_slot": MAX_POTION_SLOT,
        "max_target_slot": MAX_TARGET_SLOT,
    }
