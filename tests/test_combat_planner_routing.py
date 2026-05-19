"""Combat routing: planner-first vs policy-first when --policy is enabled."""

from __future__ import annotations

from unittest.mock import patch

from sts2_agent.agent import configure_policy, decide


def _combat_state() -> dict:
    return {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"name": "Slime", "hp": 10, "entity_id": "e1"}],
        },
        "player": {
            "energy": 3,
            "hand": [
                {"index": 0, "name": "Strike", "cost": 1, "damage": 6, "type": "attack", "can_play": True},
            ],
        },
        "run": {"floor": 1},
    }


def teardown_function() -> None:
    configure_policy(enabled=False, combat_only=False, no_combat_policy=True)


@patch("sts2_agent.agent.combat.decide_combat_potion", return_value=(None, []))
@patch("sts2_agent.agent.combat.decide_combat")
def test_planner_first_when_no_combat_policy(mock_decide_combat, _mock_potion) -> None:
    configure_policy(enabled=True, no_combat_policy=True)
    mock_decide_combat.return_value = (
        {"action": "play_card", "card_index": 0},
        ["lethal available - skip block-first", "play decision (lethal - kill attacker): incoming=0"],
    )

    decision = decide(_combat_state())

    assert decision.action == {"action": "play_card", "card_index": 0}
    assert any(r == "planner: lethal" for r in decision.reasons)
    assert not any("policy_net" in r for r in decision.reasons)
    mock_decide_combat.assert_called_once()


@patch("sts2_agent.agent.combat.decide_combat_potion", return_value=(None, []))
@patch("sts2_agent.agent._decide_policy")
@patch("sts2_agent.agent.combat.decide_combat")
def test_planner_abstain_falls_back_to_policy(
    mock_decide_combat,
    mock_policy,
    _mock_potion,
) -> None:
    from sts2_agent.agent_types import Decision

    configure_policy(enabled=True, no_combat_policy=True)
    mock_decide_combat.return_value = (None, ["waiting - not player turn"])
    mock_policy.return_value = Decision(
        {"action": "end_turn"},
        ["policy_net class=1 key=end_turn conf=90.0%"],
    )

    decision = decide(_combat_state())

    assert decision.action == {"action": "end_turn"}
    assert any("planner abstained → policy fallback" in r for r in decision.reasons)
    assert any("policy_net" in r for r in decision.reasons)
    mock_policy.assert_called_once()


@patch("sts2_agent.agent.combat.decide_combat_potion", return_value=(None, []))
@patch("sts2_agent.agent._decide_policy")
@patch("sts2_agent.agent.combat.decide_combat")
def test_legacy_policy_first_when_no_combat_policy_disabled(
    mock_decide_combat,
    mock_policy,
    _mock_potion,
) -> None:
    from sts2_agent.agent_types import Decision

    configure_policy(enabled=True, no_combat_policy=False)
    mock_policy.return_value = Decision(
        {"action": "end_turn"},
        ["policy_net class=1 key=end_turn conf=90.0%"],
    )

    decision = decide(_combat_state())

    assert decision.action == {"action": "end_turn"}
    assert any("policy_net" in r for r in decision.reasons)
    mock_decide_combat.assert_not_called()
    mock_policy.assert_called_once()


@patch("sts2_agent.agent.combat.decide_combat_potion")
def test_potion_before_planner(mock_potion) -> None:
    configure_policy(enabled=True, no_combat_policy=True)
    mock_potion.return_value = (
        {"action": "use_potion", "slot": 0},
        ["low hp heal"],
    )

    decision = decide(_combat_state())

    assert decision.action == {"action": "use_potion", "slot": 0}
    assert any("before planner" in r for r in decision.reasons)
