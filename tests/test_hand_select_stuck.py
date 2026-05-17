"""hand_select overlay - policy bypass and toggle desync."""

from __future__ import annotations

from sts2_agent.agent import configure_policy, decide
from sts2_agent.combat import (
    _hand_select_attempted,
    clear_hand_select_session,
    decide_combat,
    hand_select_session_key,
)


def _hand_select_state(**overrides) -> dict:
    cards = [
        {"index": 0, "id": "STRIKE", "name": "Strike", "can_select": True},
        {"index": 1, "id": "DEFEND", "name": "Defend", "can_select": True},
    ]
    hs = {
        "mode": "upgrade_select",
        "can_confirm": False,
        "cards": cards,
        **overrides.get("hand_select", {}),
    }
    return {
        "state_type": "hand_select",
        "hand_select": hs,
        "player": {"hand": cards},
        "run": {"floor": 3},
        **{k: v for k, v in overrides.items() if k != "hand_select"},
    }


def test_policy_skipped_on_hand_select() -> None:
    configure_policy(enabled=True, combat_only=False)
    state = _hand_select_state()
    clear_hand_select_session(state)
    d = decide(state)
    assert d.action == {"action": "combat_select_card", "card_index": 0}
    assert not any("policy_net" in r for r in d.reasons)
    configure_policy(enabled=False, combat_only=False)


def test_hand_select_advances_index_after_failed_toggle() -> None:
    state = _hand_select_state()
    clear_hand_select_session(state)
    key = hand_select_session_key(state)
    _hand_select_attempted[key] = {0}

    act, _ = decide_combat(state)
    assert act == {"action": "combat_select_card", "card_index": 1}


def test_hand_select_confirms_when_card_marked_selected() -> None:
    cards = [
        {"index": 0, "id": "STRIKE", "name": "Strike", "can_select": True, "is_selected": True},
        {"index": 1, "id": "DEFEND", "name": "Defend", "can_select": True},
    ]
    state = _hand_select_state(
        hand_select={"mode": "upgrade_select", "can_confirm": False, "cards": cards},
    )
    clear_hand_select_session(state)

    act, reasons = decide_combat(state)
    assert act == {"action": "combat_confirm_selection"}
    assert any("selected" in r for r in reasons)
