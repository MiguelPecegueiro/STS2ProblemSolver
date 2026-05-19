"""Card-reward BC specialist: masking and agent routing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sts2_agent.agent import (
    card_reward_bc_enabled,
    configure_card_reward_bc,
    configure_policy,
    configure_ppo_macro,
    decide,
)
from sts2_agent.agent_types import Decision
from training.card_reward_bc import (
    allowed_card_reward_class_ids,
    apply_card_reward_mask,
    predict_card_reward_masked,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL = REPO_ROOT / "models" / "bc_human_card.pt"
CONFIG = REPO_ROOT / "models" / "bc_human_card_config.json"
PPO_MODEL = REPO_ROOT / "models" / "ppo_v5.pt"
PPO_CONFIG = REPO_ROOT / "models" / "ppo_config.json"


def _card_reward_state(
    *,
    cards: list[dict] | None = None,
    can_skip: bool = True,
) -> dict:
    if cards is None:
        cards = [
            {"index": 0, "name": "Strike"},
            {"index": 1, "name": "Defend"},
            {"index": 2, "name": "Bash"},
        ]
    return {
        "state_type": "card_reward",
        "card_reward": {"cards": cards, "can_skip": can_skip},
        "run": {"floor": 5, "act": 1},
        "player": {"hp": 50, "max_hp": 80, "gold": 99},
    }


def teardown_function() -> None:
    configure_policy(enabled=False, combat_only=False, no_combat_policy=True)
    configure_card_reward_bc(enabled=False)
    configure_ppo_macro(enabled=False)


def test_allowed_ids_only_offered_indices() -> None:
    state = _card_reward_state(
        cards=[{"index": 2, "name": "A"}, {"index": 5, "name": "B"}],
        can_skip=False,
    )
    vocab = {
        "select_card_reward:0": 0,
        "select_card_reward:1": 1,
        "select_card_reward:2": 2,
        "select_card_reward:5": 6,
        "skip_card_reward": 7,
    }
    allowed = allowed_card_reward_class_ids(state, vocab)
    assert allowed == {2, 6}


def test_allowed_ids_include_skip_when_possible() -> None:
    state = _card_reward_state(can_skip=True)
    vocab = {
        "select_card_reward:0": 0,
        "select_card_reward:1": 1,
        "select_card_reward:2": 2,
        "skip_card_reward": 7,
    }
    allowed = allowed_card_reward_class_ids(state, vocab)
    assert allowed == {0, 1, 2, 7}


def test_allowed_ids_exclude_skip_when_not_skippable() -> None:
    state = _card_reward_state(can_skip=False)
    vocab = {
        "select_card_reward:0": 0,
        "select_card_reward:1": 1,
        "select_card_reward:2": 2,
        "skip_card_reward": 7,
    }
    allowed = allowed_card_reward_class_ids(state, vocab)
    assert 7 not in allowed
    assert allowed == {0, 1, 2}


def test_mask_forces_argmax_to_legal_class() -> None:
    logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
    masked = apply_card_reward_mask(logits, {2, 7})
    assert int(torch.argmax(masked).item()) == 2

    masked_skip = apply_card_reward_mask(logits, {7})
    assert int(torch.argmax(masked_skip).item()) == 7


@patch("training.card_reward_bc.get_card_reward_policy")
def test_predict_masked_never_picks_illegal_index(mock_get_policy) -> None:
    state = _card_reward_state(
        cards=[{"index": 1, "name": "Only"}],
        can_skip=True,
    )
    policy = MagicMock()
    policy.feature_dim = 195
    policy.device = torch.device("cpu")
    policy.id_to_key = {
        0: "select_card_reward:0",
        1: "select_card_reward:1",
        7: "skip_card_reward",
    }
    # Model strongly prefers illegal index 0 over legal 1 and skip.
    policy.model.return_value = torch.tensor([[100.0, 5.0, 0, 0, 0, 0, 0, 1.0]])
    policy.encode_state.return_value = torch.zeros(195).numpy()
    mock_get_policy.return_value = policy

    action, _reasons = predict_card_reward_masked(policy, state)
    assert action == {"action": "select_card_reward", "card_index": 1}


@patch("sts2_agent.agent.fetch_macro_qwen_context")
@patch("training.card_reward_bc.get_card_reward_policy")
def test_decide_skips_qwen_when_card_bc_enabled(
    mock_get_policy,
    mock_qwen_ctx,
) -> None:
    configure_card_reward_bc(enabled=True, model_path=MODEL, config_path=CONFIG)
    assert card_reward_bc_enabled()

    policy = MagicMock()
    policy.feature_dim = 195
    policy.device = torch.device("cpu")
    policy.id_to_key = {1: "select_card_reward:1", 7: "skip_card_reward"}
    policy.model.return_value = torch.tensor([[0.0, 5.0, 0, 0, 0, 0, 0, 0.0]])
    policy.encode_state.return_value = torch.zeros(195).numpy()
    mock_get_policy.return_value = policy

    state = _card_reward_state(cards=[{"index": 1, "name": "Defend"}], can_skip=False)
    decision = decide(state)

    mock_qwen_ctx.assert_not_called()
    assert decision.action == {"action": "select_card_reward", "card_index": 1}
    assert any("card_reward_bc" in r for r in decision.reasons)


@patch("sts2_agent.rewards.decide_card_reward")
@patch("training.card_reward_bc.get_card_reward_policy")
def test_decide_rules_fallback_when_bc_invalid(
    mock_get_policy,
    mock_rules,
) -> None:
    configure_card_reward_bc(enabled=True, model_path=MODEL, config_path=CONFIG)
    mock_get_policy.side_effect = RuntimeError("missing weights")

    mock_rules.return_value = (
        {"action": "skip_card_reward"},
        ["rules: score too low"],
    )
    state = _card_reward_state()
    decision = decide(state)

    assert decision.action == {"action": "skip_card_reward"}
    assert any("→ rules" in r for r in decision.reasons)
    mock_rules.assert_called_once()


@patch("sts2_agent.agent._decide_ppo_macro")
@patch("sts2_agent.agent._decide_card_reward_bc")
def test_map_uses_ppo_when_card_bc_enabled(mock_card_bc, mock_ppo) -> None:
    configure_card_reward_bc(enabled=True, model_path=MODEL, config_path=CONFIG)
    configure_ppo_macro(enabled=True, model_path=PPO_MODEL, config_path=PPO_CONFIG)
    mock_ppo.return_value = Decision(
        {"action": "choose_map_node", "index": 0},
        ["ppo_macro"],
    )

    state = {
        "state_type": "map",
        "map": {"choices": [{"index": 0, "type": "monster"}]},
        "run": {"floor": 1},
    }
    decision = decide(state)

    mock_card_bc.assert_not_called()
    mock_ppo.assert_called_once()
    assert decision.action == {"action": "choose_map_node", "index": 0}
