"""Load policy_net.pt and predict actions from live game state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.actions import decode_action
from training.features import encode_snapshot
from training.model import PolicyNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BC_MODEL_PATH = PROJECT_ROOT / "models" / "policy_net.pt"
DEFAULT_BC_CONFIG_PATH = PROJECT_ROOT / "models" / "model_config.json"
DEFAULT_PPO_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_v1.pt"
DEFAULT_PPO_CONFIG_PATH = PROJECT_ROOT / "models" / "ppo_config.json"
DEFAULT_MODEL_PATH = DEFAULT_BC_MODEL_PATH
DEFAULT_CONFIG_PATH = DEFAULT_BC_CONFIG_PATH


def resolve_model_paths(
    model_path: Path | str | None = None,
    config_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Prefer PPO checkpoint when present unless paths are explicit."""
    if model_path is not None and config_path is not None:
        return Path(model_path), Path(config_path)
    if DEFAULT_PPO_MODEL_PATH.exists():
        cfg = DEFAULT_PPO_CONFIG_PATH if DEFAULT_PPO_CONFIG_PATH.exists() else DEFAULT_BC_CONFIG_PATH
        return DEFAULT_PPO_MODEL_PATH, cfg
    return DEFAULT_BC_MODEL_PATH, DEFAULT_BC_CONFIG_PATH


def _living_enemies(state: dict) -> list[dict]:
    battle = state.get("battle") or {}
    out: list[dict] = []
    for enemy in battle.get("enemies") or []:
        if isinstance(enemy, dict) and int(enemy.get("hp") or 0) > 0:
            out.append(enemy)
    return out


def _resolve_action_targets(action: dict[str, Any], state: dict) -> dict[str, Any]:
    """Map ENEMY_{slot} placeholders from training decode to live entity_id."""
    target = action.get("target")
    if target is None:
        return action
    text = str(target)
    if not text.startswith("ENEMY_"):
        return action
    try:
        slot = int(text.rsplit("_", 1)[-1])
    except ValueError:
        return action
    living = _living_enemies(state)
    if 0 <= slot < len(living):
        entity = living[slot].get("entity_id") or living[slot].get("id")
        if entity:
            action = dict(action)
            action["target"] = entity
    return action


def snapshot_from_state(state: dict) -> dict[str, Any]:
    """Build the same snapshot shape used in decisions.jsonl (for encoding)."""
    from sts2_agent.data_pipeline import build_state_snapshot

    return build_state_snapshot(state)


class PolicyModel:
    """Behavioral-cloning policy loaded from models/policy_net.pt."""

    def __init__(
        self,
        model: PolicyNet,
        id_to_key: dict[int, str],
        *,
        device: torch.device,
        feature_dim: int,
    ) -> None:
        self.model = model
        self.id_to_key = id_to_key
        self.device = device
        self.feature_dim = feature_dim

    @classmethod
    def load(
        cls,
        model_path: Path | str | None = None,
        config_path: Path | str | None = None,
        *,
        device_name: str = "cpu",
    ) -> PolicyModel:
        model_path, config_path = resolve_model_paths(model_path, config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        vocab = config["action_vocab"]["action_to_id"]
        id_to_key = {int(k): v for k, v in config["action_vocab"]["id_to_action_key"].items()}
        feature_dim = int(config["feature_layout"]["feature_dim"])

        device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = PolicyNet(
            input_dim=int(checkpoint.get("input_dim", feature_dim)),
            num_actions=int(checkpoint.get("num_actions", len(vocab))),
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return cls(model, id_to_key, device=device, feature_dim=feature_dim)

    def encode_state(self, state: dict) -> np.ndarray:
        snap = snapshot_from_state(state)
        run = state.get("run") or {}
        return encode_snapshot(
            snap,
            state_type=str(state.get("state_type") or ""),
            floor=int(run.get("floor") or 0),
            act=int(run.get("act") or 1),
            immediate_reward=0.0,
        )

    @torch.no_grad()
    def predict(
        self,
        state: dict,
        *,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Return (action_dict, reasoning) for a live API state."""
        x = self.encode_state(state)
        if x.shape[0] != self.feature_dim:
            return None, [f"feature dim mismatch: got {x.shape[0]}, expected {self.feature_dim}"]

        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
        logits = self.model(xt)[0]
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=0)

        if temperature <= 0:
            class_id = int(torch.argmax(probs).item())
        else:
            class_id = int(torch.multinomial(probs, 1).item())

        action = decode_action(class_id, self.id_to_key)
        action = _resolve_action_targets(action, state)

        key = self.id_to_key.get(class_id, "?")
        conf = float(probs[class_id].item())
        top3 = torch.topk(probs, k=min(3, len(probs)))
        reasons = [
            f"policy_net class={class_id} key={key} conf={conf:.1%}",
        ]
        for prob, idx in zip(top3.values.tolist(), top3.indices.tolist()):
            reasons.append(f"  alt: {self.id_to_key.get(int(idx), '?')} ({prob:.1%})")

        return action, reasons


_policy_singleton: PolicyModel | None = None


def get_policy(
    model_path: Path | str | None = None,
    config_path: Path | str | None = None,
    *,
    reload: bool = False,
) -> PolicyModel:
    global _policy_singleton
    resolved_model, resolved_config = resolve_model_paths(model_path, config_path)
    if _policy_singleton is None or reload:
        _policy_singleton = PolicyModel.load(resolved_model, resolved_config)
    return _policy_singleton
