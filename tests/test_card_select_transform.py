"""Card-select overlay logic (transform / multi-pick events)."""

from __future__ import annotations

from sts2_agent.card_select import (
    card_select_session_key,
    clear_card_select_session,
    decide_card_select,
    effective_required_count,
    effective_selected_indices,
    sync_card_select_after_action,
)
from sts2_agent.state_parse import get_card_select_screen


def _transform_screen(prompt: str = "Choose 2 cards to Transform.", **overrides) -> dict:
    cards = [
        {"index": i, "id": f"CARD_{i}", "name": f"Card{i}", "can_select": True}
        for i in range(5)
    ]
    screen = {
        "screen_type": "transform",
        "prompt": prompt,
        "cards": cards,
        "preview_showing": False,
        "can_confirm": False,
        **overrides,
    }
    return {"state_type": "card_select", "card_select": screen}


def test_required_parsed_from_transform_prompt() -> None:
    screen = get_card_select_screen(_transform_screen()) or {}
    assert effective_required_count(screen) == 2


def test_picks_two_then_confirms_not_before() -> None:
    clear_card_select_session(get_card_select_screen(_transform_screen()) or {})
    state = _transform_screen()

    act1, _ = decide_card_select(state)
    assert act1 == {"action": "select_card", "index": 0}
    sync_card_select_after_action(state, state, act1)

    screen = get_card_select_screen(state) or {}
    assert 0 in effective_selected_indices(screen)

    state2 = _transform_screen(can_confirm=True, preview_showing=True)
    act2, _ = decide_card_select(state2)
    assert act2 == {"action": "select_card", "index": 1}
    sync_card_select_after_action(state2, state2, act2)

    state3 = _transform_screen(can_confirm=True, preview_showing=True)
    act3, reasons3 = decide_card_select(state3)
    assert act3 == {"action": "confirm_selection"}
    assert any("confirm" in r for r in reasons3)


def test_post_without_card_select_keeps_pick_then_confirms() -> None:
    """STS2MCP POST acks often omit card_select; session must not reset."""
    state = _transform_screen(
        prompt="Choose a card to Transform.",
        can_confirm=True,
        preview_showing=True,
    )
    clear_card_select_session(get_card_select_screen(state) or {})
    act1, _ = decide_card_select(state)
    assert act1 == {"action": "select_card", "index": 0}

    post_ack = {"status": "ok", "state_type": "combat"}
    sync_card_select_after_action(state, post_ack, act1)

    screen = get_card_select_screen(state) or {}
    assert 0 in effective_selected_indices(screen)

    act2, reasons2 = decide_card_select(state)
    assert act2 == {"action": "confirm_selection"}
    assert any("confirm" in r for r in reasons2)
    sync_card_select_after_action(state, post_ack, act2)


def test_single_transform_does_not_confirm_before_pick() -> None:
    """API often sets can_confirm/preview before any card is selected."""
    state = _transform_screen(
        prompt="Choose a card to Transform.",
        can_confirm=True,
        preview_showing=True,
    )
    clear_card_select_session(get_card_select_screen(state) or {})
    act, reasons = decide_card_select(state)
    assert act == {"action": "select_card", "index": 0}
    assert not any("single-pick / preview ready" in r for r in reasons)


def test_enchant_confirms_after_pick_not_repeat_select() -> None:
    """Enchant grid key must stay stable; confirm when can_confirm after one pick."""
    cards = [
        {"index": i, "id": "STRIKE", "name": "Strike", "can_select": True}
        for i in range(7)
    ]
    screen = {
        "screen_type": "NDeckEnchantSelectScreen",
        "prompt": "Choose a card to Enchant.",
        "cards": cards,
        "can_confirm": True,
        "preview_showing": False,
    }
    clear_card_select_session(screen)
    state = {"state_type": "card_select", "card_select": screen}

    act1, _ = decide_card_select(state)
    assert act1 == {"action": "select_card", "index": 0}
    sync_card_select_after_action(state, state, act1)

    # Preview shrinks grid - session key must not drop the selection.
    preview_screen = {
        **screen,
        "cards": [cards[0]],
        "preview_showing": True,
    }
    assert card_select_session_key(screen) == card_select_session_key(preview_screen)

    prev = {"state_type": "card_select", "card_select": screen}
    new = {"state_type": "card_select", "card_select": preview_screen}
    sync_card_select_after_action(
        prev, new, {"action": "select_card"},
    )
    act2, reasons2 = decide_card_select(new)
    assert act2 == {"action": "confirm_selection"}
    assert any("confirm" in r for r in reasons2)


def test_remove_does_not_repick_same_index_after_confirm() -> None:
    """Card removal events stay on the same screen for multiple confirms."""
    cards = [
        {"index": i, "id": "STRIKE_IRONCLAD", "name": "Strike", "can_select": True}
        for i in range(4)
    ]
    screen = {
        "screen_type": "select",
        "prompt": "Choose a card to Remove.",
        "cards": cards,
        "can_confirm": True,
        "preview_showing": True,
    }
    clear_card_select_session(screen)
    state = {"state_type": "card_select", "card_select": screen}

    act1, _ = decide_card_select(state)
    assert act1 == {"action": "select_card", "index": 0}
    sync_card_select_after_action(state, state, act1)

    sync_card_select_after_action(
        state,
        state,
        {"action": "confirm_selection"},
    )
    act2, _ = decide_card_select(state)
    assert act2 == {"action": "select_card", "index": 1}


def test_batch_transform_after_confirm_exits_via_confirm() -> None:
    clear_card_select_session(get_card_select_screen(_transform_screen()) or {})
    state = _transform_screen(can_confirm=True, preview_showing=True)
    sync_card_select_after_action(
        state,
        state,
        {"action": "confirm_selection"},
    )
    act, reasons = decide_card_select(state)
    assert act == {"action": "confirm_selection"}
    assert any("batch transform done" in r for r in reasons)


def test_sequential_transform_picks_new_card_after_confirm() -> None:
    prev = _transform_screen(
        prompt="Choose a card to Transform.",
        can_confirm=True,
        preview_showing=True,
    )
    still = _transform_screen(
        prompt="Choose a card to Transform.",
        can_confirm=True,
        preview_showing=False,
    )
    clear_card_select_session(get_card_select_screen(prev) or {})

    act0, _ = decide_card_select(prev)
    assert act0 == {"action": "select_card", "index": 0}
    sync_card_select_after_action(prev, prev, act0)
    sync_card_select_after_action(prev, still, {"action": "confirm_selection"})
    act, _ = decide_card_select(still)
    assert act == {"action": "select_card", "index": 1}
