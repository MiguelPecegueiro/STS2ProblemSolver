"""Morphic Grove / batch transform regressions."""

from __future__ import annotations

from sts2_agent.card_select import (
    clear_card_select_session,
    decide_card_select,
    sync_card_select_after_action,
)
from sts2_agent.state_parse import card_select_overlay_key, get_card_select_screen


def _morphic_pick_two(**overrides) -> dict:
    screen = {
        "screen_type": "transform",
        "prompt": "Choose 2 cards to Transform.",
        "required_selections": 2,
        "cards": [
            {"index": i, "id": f"C{i}", "name": f"Card{i}", "can_select": True}
            for i in range(5)
        ],
        "preview_showing": False,
        "can_confirm": False,
        **overrides,
    }
    return {"state_type": "event", "card_select": screen}


def test_batch_transform_overlay_key_stable_on_preview_prompt() -> None:
    grid = _morphic_pick_two()["card_select"]
    preview = {
        **grid,
        "prompt": "Confirm your selection.",
        "preview_showing": True,
        "can_confirm": True,
    }
    assert card_select_overlay_key(grid) == "transform:multi"
    assert card_select_overlay_key(preview) == "transform:multi"


def test_post_ack_without_card_select_keeps_multi_pick_session() -> None:
    """STS2MCP POST returns {status: ok} - session must survive for pick 2."""
    state = _morphic_pick_two()
    clear_card_select_session(get_card_select_screen(state) or {})
    act1, _ = decide_card_select(state)
    assert act1 == {"action": "select_card", "index": 0}

    post_ack = {"status": "ok", "message": "selected"}
    sync_card_select_after_action(state, post_ack, act1)

    act2, reasons2 = decide_card_select(state)
    assert act2 == {"action": "select_card", "index": 1}
    assert any("selected_indices=[0]" in r for r in reasons2)


def test_no_confirm_when_local_count_met_but_api_not_ready() -> None:
    """Stale local picks must not spam confirm when can_confirm is false."""
    state = _morphic_pick_two()
    clear_card_select_session(get_card_select_screen(state) or {})
    screen = get_card_select_screen(state) or {}
    key = card_select_overlay_key(screen)

    from sts2_agent import card_select as cs

    cs._local_selected[key] = {0, 1}
    act, _ = decide_card_select(state)
    assert act == {"action": "select_card", "index": 2}


def test_toggle_desync_advances_to_next_card() -> None:
    """If index 0 was sent but not selected, do not spam select_card 0."""
    state = _morphic_pick_two()
    clear_card_select_session(get_card_select_screen(state) or {})
    screen = get_card_select_screen(state) or {}
    key = card_select_overlay_key(screen)

    from sts2_agent import card_select as cs

    cs._attempted_toggles[key] = {0}
    act, _ = decide_card_select(state)
    assert act == {"action": "select_card", "index": 1}
