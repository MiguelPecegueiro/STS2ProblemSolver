"""Policy action validation edge cases."""

from __future__ import annotations

from sts2_agent.action_validate import normalize_policy_action, validate_policy_action


def _combat_state(hand: list[dict]) -> dict:
    return {
        "state_type": "monster",
        "battle": {
            "is_play_phase": True,
            "turn": "player",
            "enemies": [
                {
                    "hp": 20,
                    "entity_id": "ENEMY_0",
                    "name": "Slime",
                }
            ],
        },
        "player": {"energy": 3, "hand": hand},
    }


def test_play_card_with_x_cost_does_not_crash() -> None:
    state = _combat_state(
        [
            {
                "index": 0,
                "name": "Whirlwind",
                "cost": "X",
                "type": "attack",
                "can_play": True,
            }
        ]
    )
    valid, reason = validate_policy_action(
        state, {"action": "play_card", "card_index": 0, "target": "ENEMY_0"}
    )
    assert valid
    assert reason == "ok"


def test_play_card_rejects_when_cost_exceeds_energy() -> None:
    state = _combat_state(
        [{"index": 0, "name": "Bash", "cost": 2, "type": "attack", "can_play": True}]
    )
    state["player"]["energy"] = 1
    valid, reason = validate_policy_action(
        state, {"action": "play_card", "card_index": 0, "target": "ENEMY_0"}
    )
    assert not valid
    assert "cost" in reason


def _card_reward_state(*, can_skip: bool = True) -> dict:
    return {
        "state_type": "card_reward",
        "card_reward": {
            "can_skip": can_skip,
            "cards": [
                {"index": 0, "name": "Strike"},
                {"index": 1, "name": "Defend"},
            ],
        },
    }


def test_card_reward_proceed_normalizes_to_skip() -> None:
    state = _card_reward_state()
    action = normalize_policy_action(state, {"action": "proceed"})
    assert action == {"action": "skip_card_reward"}


def test_card_reward_proceed_not_valid_without_normalization() -> None:
    state = _card_reward_state()
    valid, reason = validate_policy_action(state, {"action": "proceed"})
    assert not valid
    assert "unexpected" in reason


def test_card_reward_skip_valid_when_can_skip() -> None:
    state = _card_reward_state()
    valid, reason = validate_policy_action(state, {"action": "skip_card_reward"})
    assert valid
    assert reason == "ok"


def test_end_turn_rejected_on_enemy_turn() -> None:
    state = _combat_state([])
    state["battle"]["turn"] = "enemy"
    state["battle"]["is_play_phase"] = True
    valid, reason = validate_policy_action(state, {"action": "end_turn"})
    assert not valid
    assert "player turn" in reason
