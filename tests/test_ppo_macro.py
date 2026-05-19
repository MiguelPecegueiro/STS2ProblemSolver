"""PPO macro routing and action masking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sts2_agent.agent import (
    configure_card_reward_bc,
    configure_policy,
    configure_ppo_macro,
    decide,
    ppo_macro_enabled,
)
from sts2_agent.qwen_advisor import configure_qwen
from training.ppo_macro import (
    allowed_ppo_macro_class_ids,
    apply_card_reward_mask,
    predict_ppo_macro_masked,
    ppo_macro_state_types,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PPO_MODEL = REPO_ROOT / "models" / "ppo_v5.pt"
PPO_CONFIG = REPO_ROOT / "models" / "ppo_config.json"


def teardown_function() -> None:
    configure_policy(enabled=False)
    configure_card_reward_bc(enabled=False)
    configure_ppo_macro(enabled=False)
    configure_qwen(enabled=False, macro_enabled=False, macro_context_enabled=False)


def test_ppo_macro_state_types() -> None:
    assert ppo_macro_state_types() == frozenset(
        {"map", "rest_site", "shop", "fake_merchant", "event"}
    )
    assert "card_reward" not in ppo_macro_state_types()
    assert "rewards" not in ppo_macro_state_types()


def test_map_mask_only_offered_nodes() -> None:
    state = {
        "state_type": "map",
        "map": {
            "choices": [
                {"index": 1, "room_type": "monster"},
                {"index": 3, "room_type": "elite"},
            ]
        },
    }
    vocab = {
        "choose_map_node:0": 6,
        "choose_map_node:1": 7,
        "choose_map_node:2": 8,
        "choose_map_node:3": 9,
        "end_turn": 29,
    }
    allowed = allowed_ppo_macro_class_ids(state, vocab)
    assert allowed == {7, 9}


def test_event_mask_includes_dialogue_and_options() -> None:
    state = {
        "state_type": "event",
        "event": {
            "body": "Hello",
            "options": [
                {"index": 0, "title": "Take", "is_locked": False},
                {"index": 1, "title": "Leave", "is_locked": True},
            ],
        },
    }
    vocab = {
        "advance_dialogue": 0,
        "choose_event_option:0": 2,
        "choose_event_option:1": 3,
        "proceed": 70,
    }
    allowed = allowed_ppo_macro_class_ids(state, vocab)
    assert allowed == {2, 70}


def test_mask_forces_legal_map_node() -> None:
    logits = torch.zeros(110)
    logits[6] = 100.0
    logits[7] = 1.0
    masked = apply_card_reward_mask(logits, {7})
    assert int(torch.argmax(masked).item()) == 7


@patch("training.ppo_macro.get_ppo_macro_policy")
def test_decide_uses_ppo_not_qwen_on_map(mock_get_policy) -> None:
    configure_ppo_macro(enabled=True, model_path=PPO_MODEL, config_path=PPO_CONFIG)
    configure_qwen(enabled=True, macro_enabled=False, macro_context_enabled=False)
    assert ppo_macro_enabled()

    policy = MagicMock()
    policy.feature_dim = 195
    policy.device = torch.device("cpu")
    policy.id_to_key = {7: "choose_map_node:1"}
    logits = torch.zeros(110)
    logits[6] = 100.0
    logits[7] = 5.0
    policy.model.return_value = logits
    policy.encode_state.return_value = torch.zeros(195).numpy()
    mock_get_policy.return_value = policy

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

    with patch("sts2_agent.agent.fetch_macro_qwen_context") as mock_ctx:
        mock_ctx.return_value = []
        decision = decide(state)
    assert decision.action == {"action": "choose_map_node", "index": 1}
    assert any("ppo_macro" in r for r in decision.reasons)


@patch("sts2_agent.map.decide_map")
@patch("training.ppo_macro.get_ppo_macro_policy")
def test_ppo_invalid_falls_back_to_rules(mock_get_policy, mock_rules) -> None:
    configure_ppo_macro(enabled=True, model_path=PPO_MODEL, config_path=PPO_CONFIG)
    mock_get_policy.side_effect = RuntimeError("no checkpoint")

    mock_rules.return_value = (
        {"action": "choose_map_node", "index": 0},
        ["rules: path score"],
    )
    state = {
        "state_type": "map",
        "run": {"floor": 1, "act": 1},
        "player": {"hp": 70, "max_hp": 70, "gold": 0, "relics": [], "deck": []},
        "map": {"choices": [{"index": 0, "room_type": "monster"}]},
    }
    decision = decide(state)
    assert decision.action == {"action": "choose_map_node", "index": 0}
    assert any("→ rules" in r for r in decision.reasons)
    mock_rules.assert_called_once()


@patch("sts2_agent.agent.combat.decide_combat")
def test_combat_uses_planner_not_ppo_macro(mock_combat) -> None:
    configure_ppo_macro(enabled=True, model_path=PPO_MODEL, config_path=PPO_CONFIG)
    mock_combat.return_value = (
        {"action": "end_turn"},
        ["end turn"],
    )
    state = {
        "state_type": "monster",
        "battle": {
            "turn": "player",
            "is_play_phase": True,
            "enemies": [{"name": "Slime", "hp": 10, "entity_id": "e1"}],
        },
        "player": {"energy": 3, "hand": []},
        "run": {"floor": 1},
    }
    with patch("sts2_agent.agent.combat.decide_combat_potion", return_value=(None, [])):
        decision = decide(state)
    assert decision.action == {"action": "end_turn"}
    mock_combat.assert_called()
