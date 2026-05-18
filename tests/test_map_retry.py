"""Map node selection - skip failed indices."""

from __future__ import annotations

from sts2_agent.map import decide_map, mark_map_choice_failed


def _map_state() -> dict:
    return {
        "state_type": "map",
        "run": {"floor": 5, "act": 1},
        "player": {"hp": 50, "max_hp": 80, "gold": 100},
        "map": {
            "next_options": [
                {"index": 0, "type": "Monster"},
                {"index": 1, "type": "RestSite"},
            ],
        },
    }


def test_map_skips_failed_index() -> None:
    state = _map_state()
    mark_map_choice_failed(state, 0)
    action, reasons = decide_map(state)
    assert action == {"action": "choose_map_node", "index": 1}
    assert any("blocked indices" in r for r in reasons)
