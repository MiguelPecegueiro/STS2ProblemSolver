"""card_select confirm loop when API state does not advance."""

from __future__ import annotations

from sts2_agent.card_select import (
    _confirm_stall_count,
    _last_confirm_snapshot,
    clear_card_select_session,
    decide_card_select,
    effective_selected_indices,
    sync_card_select_after_action,
)
from sts2_agent.state_parse import get_card_select_screen


def _pick_three_screen(**overrides) -> dict:
    screen = {
        "screen_type": "multiselect",
        "prompt": "Choose 3 cards.",
        "required_selections": 3,
        "cards": [
            {"index": i, "id": f"C{i}", "name": f"Card{i}", "can_select": True}
            for i in range(5)
        ],
        "can_confirm": True,
        "preview_showing": True,
        **overrides,
    }
    return {"state_type": "event", "card_select": screen}


def test_confirm_stall_stops_spamming_confirm() -> None:
    state = _pick_three_screen()
    screen = get_card_select_screen(state) or {}
    clear_card_select_session(screen)
    key = __import__("sts2_agent.state_parse", fromlist=["card_select_overlay_key"]).card_select_overlay_key(
        screen
    )

    from sts2_agent import card_select as cs

    cs._local_selected[key] = {0, 1, 2}
    cs._last_confirm_snapshot[key] = cs._confirm_snapshot(
        screen, effective_selected_indices(screen) | {0, 1, 2}, 3
    )
    cs._confirm_stall_count[key] = 2

    act, reasons = decide_card_select(state)
    assert act == {"action": "select_card", "index": 0}
    assert any("stalled" in r for r in reasons)


def test_confirm_records_api_selected_on_batch() -> None:
    state = _pick_three_screen(can_confirm=False, preview_showing=False)
    screen = get_card_select_screen(state) or {}
    clear_card_select_session(screen)

    picked = {
        **screen,
        "selected_cards": [{"index": 0}, {"index": 1}],
        "can_confirm": True,
        "preview_showing": True,
    }
    prev = {"state_type": "event", "card_select": screen}
    new = {"state_type": "event", "card_select": picked}
    sync_card_select_after_action(prev, new, {"action": "confirm_selection"})

    from sts2_agent.card_select import _confirmed_picks
    from sts2_agent.state_parse import card_select_overlay_key

    key = card_select_overlay_key(screen)
    assert {0, 1} <= _confirmed_picks.get(key, set())
