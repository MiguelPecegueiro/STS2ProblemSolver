"""Tests for macro-screen Qwen routing."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from sts2_agent.agent import configure_policy, decide, policy_active_for_state
from sts2_agent.qwen_advisor import configure_qwen, is_qwen_combat_enabled, is_qwen_macro_enabled
from sts2_agent.qwen_macro import pop_macro_qwen_trace, try_qwen_macro_decide


@pytest.fixture(autouse=True)
def _reset_qwen_settings():
    configure_qwen(enabled=False, combat_enabled=False, macro_enabled=False)
    configure_policy(enabled=False)
    yield
    configure_qwen(enabled=False, combat_enabled=False, macro_enabled=False)
    configure_policy(enabled=False)


def test_combat_disabled_macro_enabled_flags():
    configure_qwen(enabled=True, combat_enabled=False, macro_enabled=True)
    assert is_qwen_combat_enabled() is False
    assert is_qwen_macro_enabled() is True


@patch("sts2_agent.qwen_macro._call_qwen_api")
def test_macro_map_decision(mock_api):
    configure_qwen(enabled=True, macro_enabled=True, combat_enabled=False)
    mock_api.return_value = json.dumps(
        {"action": "choose_map_node", "index": 1, "reasoning": "elite path"}
    )
    state = {
        "state_type": "map",
        "run": {"floor": 3, "act": 1},
        "player": {"hp": 60, "max_hp": 70, "gold": 99, "relics": [], "deck": []},
        "map": {
            "choices": [
                {"index": 0, "room_type": "monster"},
                {"index": 1, "room_type": "elite"},
            ]
        },
    }
    decision = try_qwen_macro_decide(state)
    assert decision is not None
    assert decision.action == {"action": "choose_map_node", "index": 1}
    assert any("qwen_macro" in r for r in decision.reasons)
    trace = pop_macro_qwen_trace()
    assert trace is not None
    assert "Map — choose" in trace.get("user_prompt", "")
    assert trace.get("source") == "qwen"
    assert trace.get("response")


@patch("sts2_agent.qwen_macro._call_qwen_api")
def test_macro_invalid_action_falls_through(mock_api):
    configure_qwen(enabled=True, macro_enabled=True)
    mock_api.return_value = json.dumps({"action": "choose_map_node", "index": 99})
    state = {
        "state_type": "map",
        "run": {"floor": 1, "act": 1},
        "player": {"hp": 70, "max_hp": 70, "gold": 0, "relics": [], "deck": []},
        "map": {"choices": [{"index": 0, "room_type": "monster"}]},
    }
    assert try_qwen_macro_decide(state) is None


@patch("sts2_agent.qwen_advisor.requests.post")
def test_combat_qwen_skipped_when_disabled(mock_post):
    configure_qwen(enabled=True, combat_enabled=False, macro_enabled=True)
    from sts2_agent.qwen_advisor import QwenAdvisor

    mult = QwenAdvisor().begin_fight(
        {"run": {"floor": 1}, "player": {"hp": 70, "max_hp": 70, "deck": [], "hand": []}},
        combat_type="monster",
        enemy_names=["Slime"],
    )
    mock_post.assert_not_called()
    assert mult.source == "default"


@patch("sts2_agent.agent._decide_ppo_macro")
def test_decide_uses_ppo_not_qwen_on_rest(mock_ppo):
    from sts2_agent.agent import configure_ppo_macro

    configure_qwen(enabled=True, macro_enabled=True, combat_enabled=False)
    configure_ppo_macro(enabled=True)
    mock_ppo.return_value = __import__(
        "sts2_agent.agent_types", fromlist=["Decision"]
    ).Decision({"action": "proceed"}, ["ppo_macro"])
    state = {
        "state_type": "rest_site",
        "run": {"floor": 5, "act": 1},
        "player": {"hp": 50, "max_hp": 70, "gold": 50, "relics": [], "deck": []},
        "rest_site": {"options": [{"id": "rest", "is_enabled": True}]},
    }
    decision = decide(state)
    assert decision.action == {"action": "proceed"}
    mock_ppo.assert_called_once()


def test_policy_active_on_map_when_ppo_macro_enabled():
    from sts2_agent.agent import configure_ppo_macro

    configure_ppo_macro(enabled=True)
    configure_policy(enabled=True)
    assert policy_active_for_state("map") is True
    assert policy_active_for_state("monster") is True
