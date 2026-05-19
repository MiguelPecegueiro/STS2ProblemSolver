"""PPO policy for map / shop / rest / event (macro learning domain)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sts2_agent.state_parse import (
    event_in_dialogue,
    event_option_index,
    extract_event_options,
    extract_map_choices,
    extract_rest_options,
    extract_shop_items,
    get_shop_screen,
    map_choice_index,
    rest_can_proceed,
    rest_option_index,
    shop_item_index,
)
from training.actions import decode_action
from training.card_reward_bc import apply_card_reward_mask
from training.inference import PolicyModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PPO_CONFIG = PROJECT_ROOT / "models" / "ppo_config.json"
PPO_MACRO_CHECKPOINT_CANDIDATES = (
    "ppo_v5.pt",
    "ppo_v4.pt",
    "ppo_v3.pt",
    "ppo_v1.pt",
)

PPO_MACRO_STATE_TYPES = frozenset(
    {
        "map",
        "rest_site",
        "shop",
        "fake_merchant",
        "event",
    }
)

_ppo_macro_policy: PolicyModel | None = None
_resolved_model: Path | None = None
_resolved_config: Path | None = None


def ppo_macro_state_types() -> frozenset[str]:
    return PPO_MACRO_STATE_TYPES


def resolve_ppo_macro_paths(
    model_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> tuple[Path, Path]:
    if model_path is not None and config_path is not None:
        return Path(model_path), Path(config_path)
    if model_path is not None:
        cfg = Path(config_path) if config_path else DEFAULT_PPO_CONFIG
        return Path(model_path), cfg
    for name in PPO_MACRO_CHECKPOINT_CANDIDATES:
        candidate = PROJECT_ROOT / "models" / name
        if candidate.exists():
            return candidate, DEFAULT_PPO_CONFIG
    fallback = PROJECT_ROOT / "models" / "policy_net.pt"
    cfg = PROJECT_ROOT / "models" / "model_config.json"
    return fallback, cfg


def allowed_ppo_macro_class_ids(
    state: dict,
    action_to_id: dict[str, int],
) -> set[int]:
    """Legal action classes for the current macro screen."""
    state_type = str(state.get("state_type") or "").lower()
    allowed: set[int] = set()

    def _add(key: str) -> None:
        cid = action_to_id.get(key)
        if cid is not None:
            allowed.add(int(cid))

    if state_type == "map":
        for fallback, choice in enumerate(extract_map_choices(state)):
            idx = map_choice_index(choice, fallback)
            _add(f"choose_map_node:{idx}")

    elif state_type == "rest_site":
        for fallback, option in enumerate(extract_rest_options(state)):
            if option.get("is_enabled", True):
                idx = rest_option_index(option, fallback)
                _add(f"choose_rest_option:{idx}")
        if rest_can_proceed(state):
            _add("proceed")

    elif state_type in ("shop", "fake_merchant"):
        for fallback, item in enumerate(extract_shop_items(state)):
            idx = shop_item_index(item, fallback)
            _add(f"shop_purchase:{idx}")
        screen = get_shop_screen(state) or {}
        if screen.get("can_proceed") and not extract_shop_items(state):
            _add("proceed")

    elif state_type == "event":
        if event_in_dialogue(state) or not extract_event_options(state):
            _add("advance_dialogue")
        for fallback, option in enumerate(extract_event_options(state)):
            if option.get("is_locked"):
                continue
            idx = event_option_index(option, fallback)
            _add(f"choose_event_option:{idx}")
        _add("proceed")

    return allowed


def get_ppo_macro_policy(
    model_path: Path | str | None = None,
    config_path: Path | str | None = None,
    *,
    reload: bool = False,
    device_name: str = "cpu",
) -> PolicyModel:
    global _ppo_macro_policy, _resolved_model, _resolved_config

    model, config = resolve_ppo_macro_paths(model_path, config_path)
    if (
        _ppo_macro_policy is None
        or reload
        or model != _resolved_model
        or config != _resolved_config
    ):
        _resolved_model = model
        _resolved_config = config
        _ppo_macro_policy = PolicyModel.load(model, config, device_name=device_name)

    return _ppo_macro_policy


def action_to_id_from_policy(policy: PolicyModel) -> dict[str, int]:
    return {key: int(class_id) for class_id, key in policy.id_to_key.items()}


@torch.no_grad()
def predict_ppo_macro_masked(
    policy: PolicyModel,
    state: dict,
    *,
    temperature: float = 0.0,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Masked PPO prediction for map / shop / rest / event."""
    x = policy.encode_state(state)
    if x.shape[0] != policy.feature_dim:
        return None, [
            f"feature dim mismatch: got {x.shape[0]}, expected {policy.feature_dim}"
        ]

    xt = torch.from_numpy(x).unsqueeze(0).to(policy.device)
    logits = policy.model(xt)[0]
    vocab = action_to_id_from_policy(policy)
    allowed = allowed_ppo_macro_class_ids(state, vocab)

    if not allowed:
        return None, ["ppo_macro: no legal actions on screen"]

    masked = apply_card_reward_mask(logits, allowed)

    if temperature <= 0:
        class_id = int(torch.argmax(masked).item())
    else:
        probs = torch.softmax(masked / max(temperature, 1e-6), dim=0)
        class_id = int(torch.multinomial(probs, 1).item())

    action = decode_action(class_id, policy.id_to_key)
    key = policy.id_to_key.get(class_id, "?")
    probs = torch.softmax(masked / max(temperature, 1e-6), dim=0)
    conf = float(probs[class_id].item())
    allowed_keys = sorted(policy.id_to_key.get(i, "?") for i in allowed)
    reasons = [
        "ppo_macro",
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
