"""Event screen - post-choice dialogue and proceed handling."""

from __future__ import annotations

from sts2_agent.event import decide_event, mark_event_option_failed
from sts2_agent.state_parse import event_in_dialogue


def _event_state(**screen_overrides) -> dict:
    screen = {
        "event_name": "Test Event",
        "body": "You find a strange altar.",
        "options": [
            {
                "index": 0,
                "title": "Pray",
                "was_chosen": True,
            },
        ],
        **screen_overrides,
    }
    return {"state_type": "event", "event": screen, "player": {"hp": 50, "max_hp": 80}}


def test_after_choice_advances_dialogue_not_repick() -> None:
    state = _event_state()
    action, reasons = decide_event(state)
    assert action == {"action": "advance_dialogue"}
    assert any("choice already made" in r for r in reasons)


def test_proceed_after_all_choices() -> None:
    state = _event_state(
        options=[
            {"index": 0, "title": "Pray", "was_chosen": True},
            {"index": 1, "title": "Proceed", "is_proceed": True},
        ],
    )
    action, reasons = decide_event(state)
    assert action == {"action": "choose_event_option", "index": 1}
    assert any("proceed" in r.lower() for r in reasons)


def test_blocked_option_tries_advance() -> None:
    state = _event_state(
        options=[
            {"index": 0, "title": "Pray", "was_chosen": False},
            {"index": 1, "title": "Smash", "was_chosen": False},
        ],
    )
    mark_event_option_failed(state, 0)
    action, _ = decide_event(state)
    assert action == {"action": "choose_event_option", "index": 1}


def test_in_dialogue_flag() -> None:
    state = _event_state(in_dialogue=True, options=[])
    assert event_in_dialogue(state)
    action, _ = decide_event(state)
    assert action == {"action": "advance_dialogue"}
