"""Policy must not bypass card_select session logic."""

from __future__ import annotations

from sts2_agent.agent import configure_policy, decide
from sts2_agent.card_select import clear_card_select_session, sync_card_select_after_action


def test_policy_skipped_when_card_select_overlay_active() -> None:
    configure_policy(enabled=True, combat_only=False)
    screen = {
        "screen_type": "transform",
        "prompt": "Choose 2 cards to Transform.",
        "cards": [
            {"index": i, "id": f"C{i}", "name": f"Card{i}", "can_select": True}
            for i in range(4)
        ],
        "can_confirm": False,
        "preview_showing": False,
    }
    clear_card_select_session(screen)
    state = {"state_type": "event", "card_select": screen}

    d1 = decide(state)
    assert d1.action == {"action": "select_card", "index": 0}
    assert not any("policy_net" in r for r in d1.reasons)

    sync_card_select_after_action(state, state, d1.action)
    state2 = {
        **state,
        "card_select": {**screen, "can_confirm": True, "preview_showing": True},
    }
    d2 = decide(state2)
    assert d2.action == {"action": "select_card", "index": 1}

    configure_policy(enabled=False, combat_only=False)
