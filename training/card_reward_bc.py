"""Card-reward behavioral cloning: separate checkpoint and masked inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sts2_agent.state_parse import (
    card_reward_can_skip,
    card_reward_index,
    extract_card_reward_cards,
)
from training.actions import decode_action
from training.inference import PolicyModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARD_REWARD_MODEL = PROJECT_ROOT / "models" / "bc_human_card.pt"
DEFAULT_CARD_REWARD_CONFIG = PROJECT_ROOT / "models" / "bc_human_card_config.json"

_card_reward_policy: PolicyModel | None = None
_resolved_model: Path | None = None
_resolved_config: Path | None = None


def allowed_card_reward_class_ids(
    state: dict,
    action_to_id: dict[str, int],
) -> set[int]:
    """Class ids legal on this screen: offered card indices and optional skip."""
    allowed: set[int] = set()
    for fallback, card in enumerate(extract_card_reward_cards(state)):
        api_index = card_reward_index(card, fallback)
        key = f"select_card_reward:{api_index}"
        if key in action_to_id:
            allowed.add(int(action_to_id[key]))
    if card_reward_can_skip(state):
        skip_id = action_to_id.get("skip_card_reward")
        if skip_id is not None:
            allowed.add(int(skip_id))
    return allowed


def apply_card_reward_mask(
    logits: torch.Tensor,
    allowed_ids: set[int],
) -> torch.Tensor:
    """Set disallowed action logits to -inf before softmax/argmax."""
    if not allowed_ids:
        return logits
    masked = logits.clone()
    disallowed = [i for i in range(masked.numel()) if i not in allowed_ids]
    if disallowed:
        masked[disallowed] = float("-inf")
    return masked


def get_card_reward_policy(
    model_path: Path | str | None = None,
    config_path: Path | str | None = None,
    *,
    reload: bool = False,
    device_name: str = "cpu",
) -> PolicyModel:
    """Load the card-reward specialist (not the main combat/macro policy)."""
    global _card_reward_policy, _resolved_model, _resolved_config

    model = Path(model_path) if model_path is not None else DEFAULT_CARD_REWARD_MODEL
    config = Path(config_path) if config_path is not None else DEFAULT_CARD_REWARD_CONFIG

    if (
        _card_reward_policy is None
        or reload
        or model != _resolved_model
        or config != _resolved_config
    ):
        _resolved_model = model
        _resolved_config = config
        _card_reward_policy = PolicyModel.load(model, config, device_name=device_name)

    return _card_reward_policy


def action_to_id_from_policy(policy: PolicyModel) -> dict[str, int]:
    return {key: int(class_id) for class_id, key in policy.id_to_key.items()}


@torch.no_grad()
def predict_card_reward_masked(
    policy: PolicyModel,
    state: dict,
    *,
    temperature: float = 0.0,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Predict card_reward action with masking; returns (action, reasons)."""
    x = policy.encode_state(state)
    if x.shape[0] != policy.feature_dim:
        return None, [
            f"feature dim mismatch: got {x.shape[0]}, expected {policy.feature_dim}"
        ]

    xt = torch.from_numpy(x).unsqueeze(0).to(policy.device)
    logits = policy.model(xt)[0]
    vocab = action_to_id_from_policy(policy)
    allowed = allowed_card_reward_class_ids(state, vocab)

    if not allowed:
        return None, ["card_reward_bc: no legal actions on screen"]

    masked = apply_card_reward_mask(logits, allowed)
    probs = torch.softmax(masked / max(temperature, 1e-6), dim=0)

    if temperature <= 0:
        class_id = int(torch.argmax(probs).item())
    else:
        class_id = int(torch.multinomial(probs, 1).item())

    action = decode_action(class_id, policy.id_to_key)
    key = policy.id_to_key.get(class_id, "?")
    conf = float(probs[class_id].item())
    allowed_keys = sorted(policy.id_to_key.get(i, "?") for i in allowed)
    reasons = [
        "card_reward_bc",
        f"class={class_id} key={key} conf={conf:.1%}",
        f"allowed={allowed_keys}",
    ]
    top3 = torch.topk(probs, k=min(3, len(probs)))
    for prob, idx in zip(top3.values.tolist(), top3.indices.tolist()):
        idx_int = int(idx)
        if idx_int in allowed:
            reasons.append(
                f"  alt: {policy.id_to_key.get(idx_int, '?')} ({prob:.1%})"
            )

    return action, reasons
