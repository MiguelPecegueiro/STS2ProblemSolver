"""Map node selection using scorer path heuristics.

Training data: map choices recorded via data_pipeline (main.py).
Rest-site path selection (+3) rewarded in data_pipeline.
"""

from __future__ import annotations

from sts2_agent.data_pipeline import record_handler_decision
from sts2_agent.scorer import score_map_room
from sts2_agent.state_parse import extract_map_choices, map_choice_index, map_choice_room_type

_failed_map_indices: dict[str, set[int]] = {}


def map_session_key(state: dict) -> str:
    choices = extract_map_choices(state)
    floor = int((state.get("run") or {}).get("floor") or 0)
    sig = "|".join(
        f"{map_choice_index(c, i)}:{map_choice_room_type(c)}"
        for i, c in enumerate(choices)
    )
    return f"floor{floor}|{sig}"


def mark_map_choice_failed(state: dict, index: int) -> None:
    _failed_map_indices.setdefault(map_session_key(state), set()).add(int(index))


def clear_map_session(state: dict) -> None:
    _failed_map_indices.pop(map_session_key(state), None)


def record_training(state: dict, action: dict | None, reasoning: list[str]) -> None:
    record_handler_decision(state, action, reasoning, handler="map")


def decide_map(state: dict) -> tuple[dict | None, list[str]]:
    choices = extract_map_choices(state)
    if not choices:
        return None, [
            "map - no travelable nodes in map.next_options, waiting for map screen"
        ]

    player = state.get("player") or {}
    run = state.get("run") or {}
    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    hp_ratio = hp / max_hp if max_hp else 1.0
    gold = int(player.get("gold") or 0)
    floor = int(run.get("floor") or 0)
    act = int(run.get("act") or 1)
    boss_soon = floor >= 45 or (act >= 1 and floor % 15 > 12)

    session = map_session_key(state)
    failed = _failed_map_indices.get(session, set())
    scored: list[tuple[float, int, list[str], str]] = []
    all_reasons: list[str] = [
        f"context: HP {hp}/{max_hp} ({hp_ratio:.0%}), gold {gold}, boss_soon={boss_soon}",
        f"choosable paths: {len(choices)} (from map.next_options)",
    ]
    if failed:
        all_reasons.append(f"map: blocked indices {sorted(failed)}")

    for list_idx, option in enumerate(choices):
        room = map_choice_room_type(option)
        api_index = map_choice_index(option, list_idx)
        if api_index in failed:
            continue
        result = score_map_room(room, hp_ratio=hp_ratio, gold=gold, boss_soon=boss_soon)
        scored.append((result.score, api_index, result.reasons, room))
        all_reasons.append(
            f"  option[{api_index}] {room}: {result.score:.1f} - "
            f"{', '.join(result.reasons[-2:])}"
        )

    if not scored and failed:
        _failed_map_indices.pop(session, None)
        all_reasons.append("map: cleared blocked indices - retry")
        for list_idx, option in enumerate(choices):
            room = map_choice_room_type(option)
            api_index = map_choice_index(option, list_idx)
            result = score_map_room(room, hp_ratio=hp_ratio, gold=gold, boss_soon=boss_soon)
            scored.append((result.score, api_index, result.reasons, room))

    if not scored:
        return None, all_reasons + ["map - no travelable nodes after filtering"]

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_index, best_reasons, best_room = scored[0]
    all_reasons.append(f"choose_map_node index {best_index} ({best_room}, score {best_score:.1f})")
    all_reasons.extend(best_reasons)
    return {"action": "choose_map_node", "index": best_index}, all_reasons
