"""Deck/card selection overlays (rest smith, transforms, events, etc.)."""

from __future__ import annotations

from sts2_agent.data_pipeline import record_handler_decision
from sts2_agent.knowledge import get_knowledge
from sts2_agent.scorer import card_name, card_remove_priority, smith_upgrade_priority
from sts2_agent.state_parse import (
    _selection_count_from_prompt,
    card_select_grid_index,
    card_select_is_transform,
    card_select_overlay_key,
    card_select_required_count,
    card_select_selected_indices,
    card_select_single_pick_confirm,
    get_card_select_screen,
    is_card_select_active,
)

# API often omits selected_indices until confirm - remember toggles we sent this screen.
_local_selected: dict[str, set[int]] = {}
# Completed confirm batches on this overlay (e.g. Morphic Grove pick-2).
_transform_batches_done: dict[str, int] = {}
_pending_confirm: dict[str, bool] = {}
# Indices already confirmed on sequential overlays (remove / one-at-a-time transform).
_confirmed_picks: dict[str, set[int]] = {}
# Locked pick count for transform:multi (prompt/API may drop "2" during preview).
_multi_transform_required: int = 2
# Grid indices we already sent select_card for but are not selected (toggle / POST desync).
_attempted_toggles: dict[str, set[int]] = {}
# Same confirm on unchanged screen - API accepted but did not advance.
_last_confirm_snapshot: dict[str, str] = {}
_confirm_stall_count: dict[str, int] = {}


def record_training(state: dict, action: dict | None, reasoning: list[str]) -> None:
    record_handler_decision(state, action, reasoning, handler="card_select")


def card_select_session_key(screen: dict) -> str:
    """Alias for overlay session key (stable when the card grid changes)."""
    return card_select_overlay_key(screen)


def _resolve_select_index(screen: dict, raw_index: int) -> int:
    """Map select_card list index to grid index stored in session/API."""
    cards = screen.get("cards") or []
    if 0 <= raw_index < len(cards) and isinstance(cards[raw_index], dict):
        return card_select_grid_index(cards[raw_index], raw_index)
    return int(raw_index)


def _migrate_overlay_session(old_key: str, new_key: str) -> None:
    if old_key == new_key:
        return
    for store in (
        _local_selected,
        _pending_confirm,
        _transform_batches_done,
        _confirmed_picks,
        _attempted_toggles,
    ):
        if old_key in store:
            store[new_key] = store.pop(old_key)


def effective_selected_indices(screen: dict) -> set[int]:
    selected = card_select_selected_indices(screen)
    key = card_select_overlay_key(screen)
    selected |= _local_selected.get(key, set())
    return selected


def mark_card_selected(screen: dict, grid_index: int) -> None:
    key = card_select_overlay_key(screen)
    _local_selected.setdefault(key, set()).add(int(grid_index))
    attempted = _attempted_toggles.get(key)
    if attempted is not None:
        attempted.discard(int(grid_index))


def clear_card_select_session(screen: dict) -> None:
    key = card_select_overlay_key(screen)
    _local_selected.pop(key, None)
    _pending_confirm.pop(key, None)
    _transform_batches_done.pop(key, None)
    _confirmed_picks.pop(key, None)
    _attempted_toggles.pop(key, None)
    _last_confirm_snapshot.pop(key, None)
    _confirm_stall_count.pop(key, None)
    global _multi_transform_required
    if key == "transform:multi":
        _multi_transform_required = 2


def _finished_picks(screen: dict) -> set[int]:
    return _confirmed_picks.get(card_select_overlay_key(screen), set())


def _base_required_count(screen: dict) -> int:
    need = card_select_required_count(screen)
    key = card_select_overlay_key(screen)
    if key == "transform:multi":
        global _multi_transform_required
        if need > 1:
            _multi_transform_required = max(_multi_transform_required, need)
        return max(need, _multi_transform_required)
    return need


def _prune_attempted_toggles(screen: dict, key: str) -> None:
    """Drop toggles that are now reflected in API/local selection."""
    attempted = _attempted_toggles.get(key)
    if not attempted:
        return
    selected = card_select_selected_indices(screen) | _local_selected.get(key, set())
    _attempted_toggles[key] = {idx for idx in attempted if idx not in selected}


def _is_sequential_overlay(screen: dict) -> bool:
    """One card per confirm; same screen repeats (remove, single transform)."""
    if _base_required_count(screen) > 1:
        return False
    screen_type = str(screen.get("screen_type") or "").lower()
    prompt = str(screen.get("prompt") or "").lower()
    if "remove" in prompt and screen_type == "select":
        return True
    if card_select_is_transform(screen):
        return _selection_count_from_prompt(str(screen.get("prompt") or "")) is None
    return card_select_single_pick_confirm(screen)


def effective_required_count(screen: dict) -> int:
    return _base_required_count(screen)


def _batch_transform_complete(screen: dict) -> bool:
    key = card_select_overlay_key(screen)
    need = _base_required_count(screen)
    return card_select_is_transform(screen) and need > 1 and _transform_batches_done.get(key, 0) >= 1


def _confirm_snapshot(screen: dict, selected: set[int], required: int) -> str:
    return (
        f"{required}|{sorted(selected)}|{screen.get('can_confirm')}|"
        f"{screen.get('preview_showing')}"
    )


def _confirm_stalled(key: str, screen: dict, selected: set[int], required: int) -> bool:
    snap = _confirm_snapshot(screen, selected, required)
    if _last_confirm_snapshot.get(key) == snap:
        _confirm_stall_count[key] = _confirm_stall_count.get(key, 0) + 1
    else:
        _confirm_stall_count[key] = 0
    _last_confirm_snapshot[key] = snap
    return _confirm_stall_count.get(key, 0) >= 2


def _reset_confirm_stall(key: str) -> None:
    _last_confirm_snapshot.pop(key, None)
    _confirm_stall_count.pop(key, None)
    _pending_confirm.pop(key, None)


def _record_confirm_batch(prev_screen: dict, key: str) -> None:
    api_selected = card_select_selected_indices(prev_screen)
    confirmed_now = _local_selected.pop(key, set()) | api_selected
    if _is_sequential_overlay(prev_screen):
        if confirmed_now:
            _confirmed_picks.setdefault(key, set()).update(confirmed_now)
    elif card_select_is_transform(prev_screen) and _base_required_count(prev_screen) > 1:
        _transform_batches_done[key] = _transform_batches_done.get(key, 0) + 1
        if confirmed_now:
            _confirmed_picks.setdefault(key, set()).update(confirmed_now)
    elif confirmed_now:
        _confirmed_picks.setdefault(key, set()).update(confirmed_now)
    _pending_confirm.pop(key, None)


def sync_card_select_after_action(
    prev_state: dict, new_state: dict, action: dict
) -> None:
    """Update local session after a successful POST (main loop)."""
    if not isinstance(action, dict):
        return
    name = str(action.get("action") or "")
    prev_screen = get_card_select_screen(prev_state)
    if not prev_screen:
        return
    key = card_select_overlay_key(prev_screen)

    if not is_card_select_active(new_state):
        # POST bodies often omit card_select even while the overlay is still open.
        if name == "cancel_selection":
            clear_card_select_session(prev_screen)
        elif name == "select_card":
            idx = action.get("index")
            if idx is not None:
                mark_card_selected(
                    prev_screen, _resolve_select_index(prev_screen, int(idx))
                )
            _pending_confirm.pop(key, None)
        elif name == "confirm_selection":
            _record_confirm_batch(prev_screen, key)
        return

    new_screen = get_card_select_screen(new_state) or {}
    new_key = card_select_overlay_key(new_screen)
    if new_key != key:
        _migrate_overlay_session(key, new_key)
        key = new_key

    if name == "confirm_selection":
        batch_screen = get_card_select_screen(new_state) or prev_screen
        _record_confirm_batch(batch_screen, key)
    elif name == "select_card":
        idx = action.get("index")
        if idx is not None:
            mark_card_selected(
                new_screen, _resolve_select_index(new_screen, int(idx))
            )
        _pending_confirm.pop(key, None)


def note_card_select_action_failed(state: dict, action: dict) -> None:
    """Undo optimistic state when the API rejects a card_select action."""
    screen = get_card_select_screen(state) or {}
    key = card_select_overlay_key(screen)
    name = str(action.get("action") or "")
    if name == "confirm_selection":
        _pending_confirm.pop(key, None)
        _local_selected.pop(key, None)
        return
    if name == "select_card" and action.get("index") is not None:
        grid_idx = _resolve_select_index(screen, int(action["index"]))
        _local_selected.get(key, set()).discard(grid_idx)
        _attempted_toggles.get(key, set()).discard(grid_idx)
    _pending_confirm.pop(key, None)


def api_ready_to_confirm(screen: dict, selected: set[int], required: int) -> bool:
    """True only when enough cards are selected and the game enables confirm."""
    if len(selected) < required:
        return False
    if screen.get("can_confirm"):
        return True
    if screen.get("preview_showing") and len(selected) >= required:
        return True
    return False


def _should_confirm(screen: dict, selected: set[int], required: int) -> bool:
    return api_ready_to_confirm(screen, selected, required)


def decide_card_select(state: dict) -> tuple[dict | None, list[str]]:
    screen = get_card_select_screen(state) or {}
    key = card_select_overlay_key(screen)
    _prune_attempted_toggles(screen, key)
    required = effective_required_count(screen)
    selected = effective_selected_indices(screen)
    finished = _finished_picks(screen) if _is_sequential_overlay(screen) else set()

    reasons: list[str] = [
        f"card_select screen_type={screen.get('screen_type')} "
        f"required={required} selected={len(selected)} "
        f"can_confirm={screen.get('can_confirm')} preview={screen.get('preview_showing')}"
    ]
    if _transform_batches_done.get(key):
        reasons.append(f"transform_batches_done={_transform_batches_done[key]}")
    if selected:
        reasons.append(f"selected_indices={sorted(selected)}")
    if finished:
        reasons.append(f"already_confirmed={sorted(finished)}")

    if _batch_transform_complete(screen):
        if screen.get("can_confirm") or screen.get("preview_showing"):
            if _confirm_stalled(key, screen, selected, required):
                _reset_confirm_stall(key)
                reasons.append("batch confirm stalled - retry picks")
            else:
                _pending_confirm[key] = True
                return {"action": "confirm_selection"}, reasons + [
                    "batch transform done - confirm to leave"
                ]

    if _pending_confirm.get(key) and api_ready_to_confirm(screen, selected, required):
        if _confirm_stalled(key, screen, selected, required):
            _local_selected.pop(key, None)
            _reset_confirm_stall(key)
            selected = effective_selected_indices(screen)
            reasons.append("confirm stalled - retry picks")
        else:
            _pending_confirm[key] = True
            return {"action": "confirm_selection"}, reasons + [
                f"retry confirm ({len(selected)}/{required} cards)"
            ]

    if _should_confirm(screen, selected, required):
        if _confirm_stalled(key, screen, selected, required):
            _local_selected.pop(key, None)
            _reset_confirm_stall(key)
            selected = effective_selected_indices(screen)
            reasons.append("confirm stalled - retry picks")
        else:
            _pending_confirm[key] = True
            return {"action": "confirm_selection"}, reasons + [
                f"confirm selection ({len(selected)}/{required} cards)"
            ]

    cards = screen.get("cards") or []
    if not cards:
        if screen.get("can_cancel"):
            clear_card_select_session(screen)
            return {"action": "cancel_selection"}, reasons + ["no cards - cancel"]
        return None, reasons + ["no cards visible - waiting"]

    screen_type = str(screen.get("screen_type") or "").lower()
    prompt = str(screen.get("prompt") or "").lower()
    is_upgrade = screen_type == "upgrade" or "upgrade" in prompt or "smith" in prompt
    is_remove = screen_type == "select" and "remove" in prompt

    kb = get_knowledge()
    attempted = _attempted_toggles.get(key, set())
    candidates: list[tuple[int, int, dict, float]] = []
    for list_idx, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        if card.get("can_select") is False:
            continue
        if is_upgrade and card.get("is_upgraded"):
            continue
        grid_index = card_select_grid_index(card, list_idx)
        if grid_index in selected or grid_index in finished:
            continue
        if grid_index in attempted:
            continue
        if is_upgrade:
            scored = smith_upgrade_priority(card, kb)
            score = scored.score
            reasons.append(f"  [{grid_index}] {card_name(card)}: {score:.1f}")
        elif is_remove:
            scored = card_remove_priority(card, kb)
            score = scored.score
            reasons.append(f"  [{grid_index}] {card_name(card)}: remove {score:.1f}")
        else:
            score = 40.0 - list_idx * 0.01
        candidates.append((list_idx, grid_index, card, score))

    if not candidates and attempted:
        _attempted_toggles.pop(key, None)
        reasons.append("cleared attempted toggles - retry picks")
        for list_idx, card in enumerate(cards):
            if not isinstance(card, dict) or card.get("can_select") is False:
                continue
            if is_upgrade and card.get("is_upgraded"):
                continue
            grid_index = card_select_grid_index(card, list_idx)
            if grid_index in selected or grid_index in finished:
                continue
            score = 40.0 - list_idx * 0.01
            if is_upgrade:
                scored = smith_upgrade_priority(card, kb)
                score = scored.score
            elif is_remove:
                scored = card_remove_priority(card, kb)
                score = scored.score
            candidates.append((list_idx, grid_index, card, score))

    if not candidates:
        if api_ready_to_confirm(screen, selected, required):
            if _confirm_stalled(key, screen, selected, required):
                if screen.get("can_cancel"):
                    clear_card_select_session(screen)
                    return {"action": "cancel_selection"}, reasons + [
                        "confirm stalled - cancel"
                    ]
                _reset_confirm_stall(key)
                reasons.append("confirm stalled - cleared session")
            else:
                _pending_confirm[key] = True
                return {"action": "confirm_selection"}, reasons + [
                    f"all picks selected - confirm ({len(selected)}/{required})"
                ]
        if screen.get("can_cancel"):
            clear_card_select_session(screen)
            return {"action": "cancel_selection"}, reasons + ["no selectable cards"]
        return None, reasons + ["no selectable cards - waiting"]

    candidates.sort(key=lambda x: (-x[3], x[0]))
    best_list_idx, best_grid_idx, best_card, _best_score = candidates[0]
    _attempted_toggles.setdefault(key, set()).add(best_grid_idx)
    label = card_name(best_card)
    reasons.append(
        f"select_card index {best_list_idx} ({label}) [{len(selected) + 1}/{required}]"
    )
    return {"action": "select_card", "index": best_list_idx}, reasons
