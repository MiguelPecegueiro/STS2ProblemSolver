"""Post-combat rewards flow - declined card reward loop guard."""

from __future__ import annotations

from sts2_agent.action_validate import validate_policy_action
from sts2_agent.rewards import (
    decide_rewards,
    get_rewards_flow,
    note_card_reward_claimed,
    note_card_reward_skipped,
)


def _rewards_state(*, card_index: int = 1, can_proceed: bool = True) -> dict:
    return {
        "state_type": "rewards",
        "rewards": {
            "can_proceed": can_proceed,
            "items": [
                {"index": 0, "type": "gold", "gold_amount": 25, "claimed": False},
                {
                    "index": card_index,
                    "type": "card",
                    "description": "Card reward",
                    "claimed": False,
                },
            ],
        },
        "player": {"hp": 50, "max_hp": 80, "potions": []},
    }


def setup_function() -> None:
    get_rewards_flow().clear()


def test_declined_card_reward_not_reclaimed() -> None:
    state = _rewards_state()
    note_card_reward_claimed(state, 1)
    note_card_reward_skipped()

    action, reasons = decide_rewards(state)
    assert action is not None
    assert action.get("action") == "claim_reward"
    assert action.get("index") == 0
    assert any("declined" in r.lower() for r in reasons)


def test_policy_cannot_claim_declined_card_reward() -> None:
    state = _rewards_state()
    get_rewards_flow().declined_card_reward_indices.add(1)
    valid, reason = validate_policy_action(state, {"action": "claim_reward", "index": 1})
    assert not valid
    assert "declined" in reason
