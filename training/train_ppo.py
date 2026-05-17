"""Offline PPO fine-tuning on decisions.jsonl (actor + critic)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.actions import vocab_metadata
from training.features import FEATURE_DIM, feature_layout
from training.model import PolicyNet, STS2ValueNet
from training.ppo_dataset import build_ppo_dataset, compute_gae_for_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DECISIONS = PROJECT_ROOT / "data" / "decisions.jsonl"
DEFAULT_BC_MODEL = PROJECT_ROOT / "models" / "policy_net.pt"
DEFAULT_BC_CONFIG = PROJECT_ROOT / "models" / "model_config.json"
DEFAULT_PPO_MODEL = PROJECT_ROOT / "models" / "ppo_v1.pt"
DEFAULT_PPO_CONFIG = PROJECT_ROOT / "models" / "ppo_config.json"
DEFAULT_LOG = PROJECT_ROOT / "logs" / "ppo_training.log"


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ppo_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _load_bc_actor(
    actor: PolicyNet,
    path: Path,
    *,
    logger: logging.Logger,
) -> None:
    if not path.exists():
        logger.warning("BC checkpoint not found at %s - training actor from scratch", path)
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    try:
        actor.load_state_dict(state, strict=True)
        logger.info("Loaded BC actor weights from %s", path)
    except RuntimeError as exc:
        logger.warning("Strict BC load failed (%s) - partial load", exc)
        actor.load_state_dict(state, strict=False)


@torch.no_grad()
def _evaluate_val_return(
    critic: STS2ValueNet,
    dataset,
    indices: np.ndarray,
    device: torch.device,
) -> float:
    if len(indices) == 0:
        return 0.0
    x = torch.from_numpy(dataset.X[indices]).to(device)
    values = critic(x)
    return float(values.mean().item())


def train_ppo(
    *,
    decisions_path: Path,
    start_from: Path,
    model_path: Path,
    config_path: Path,
    log_path: Path,
    epochs: int,
    batch_size: int,
    ppo_epochs: int,
    lr: float,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
    min_run_score_percentile: float,
    val_fraction: float,
    seed: int,
    device_name: str,
    terminal_reward_scale: float,
    max_grad_norm: float,
    entropy_stop_threshold: float,
) -> dict:
    logger = _setup_logging(log_path)
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab: dict[str, int] | None = None
    if DEFAULT_BC_CONFIG.exists():
        from training.ppo_dataset import load_vocab_from_config

        vocab = load_vocab_from_config(DEFAULT_BC_CONFIG)
        logger.info("Using action vocab from %s (%d actions)", DEFAULT_BC_CONFIG, len(vocab))

    dataset = build_ppo_dataset(
        decisions_path,
        gamma=gamma,
        min_run_score_percentile=min_run_score_percentile,
        val_fraction=val_fraction,
        seed=seed,
        terminal_reward_scale=terminal_reward_scale,
        vocab=vocab,
    )
    vocab = dataset.vocab
    num_actions = len(vocab)

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    logger.info(
        "PPO dataset: %d transitions, %d runs (train=%d val=%d) device=%s",
        len(dataset),
        dataset.meta["num_runs"],
        len(dataset.train_indices),
        len(dataset.val_indices),
        device,
    )
    logger.info("Reward stats: %s", dataset.meta["reward_stats"])

    actor = PolicyNet(input_dim=FEATURE_DIM, num_actions=num_actions).to(device)
    critic = STS2ValueNet(input_dim=FEATURE_DIM).to(device)
    _load_bc_actor(actor, start_from, logger=logger)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr,
    )

    train_idx = dataset.train_indices
    if len(train_idx) == 0:
        raise ValueError("No training transitions after split")

    best_val_metric = float("-inf")
    best_entropy_at_save = float("-inf")
    saved_checkpoint = False
    stop_epoch: int | None = None
    stop_reason: str | None = None
    history: list[dict] = []

    def _save_checkpoint(epoch_entropy: float, *, label: str) -> None:
        nonlocal saved_checkpoint, best_entropy_at_save
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": actor.state_dict(),
                "value_state_dict": critic.state_dict(),
                "input_dim": FEATURE_DIM,
                "num_actions": num_actions,
                "algorithm": "ppo",
                "epoch_entropy": epoch_entropy,
            },
            model_path,
        )
        saved_checkpoint = True
        best_entropy_at_save = max(best_entropy_at_save, epoch_entropy)
        logger.info(
            "%s PPO checkpoint to %s (entropy=%.4f)",
            label,
            model_path,
            epoch_entropy,
        )

    for epoch in range(1, epochs + 1):
        actor.eval()
        critic.eval()
        with torch.no_grad():
            all_values = critic(torch.from_numpy(dataset.X).to(device)).cpu().numpy()
        advantages, value_targets = compute_gae_for_dataset(
            dataset.step_rewards,
            all_values,
            dataset.run_starts,
            dataset.run_lengths,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        adv_mean = float(advantages[train_idx].mean())
        adv_std = float(advantages[train_idx].std() + 1e-8)
        advantages = (advantages - adv_mean) / adv_std

        # Snapshot old log-probs for PPO clip (full dataset, indexed per batch)
        with torch.no_grad():
            logits_all = actor(torch.from_numpy(dataset.X).to(device))
            dist_all = torch.distributions.Categorical(logits=logits_all)
            old_log_probs_all = dist_all.log_prob(
                torch.from_numpy(dataset.actions).to(device)
            ).cpu()

        actor.train()
        critic.train()

        epoch_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "batches": 0,
        }

        for ppo_pass in range(ppo_epochs):
            perm = np.random.default_rng(seed + epoch * 1000 + ppo_pass).permutation(
                len(train_idx)
            )
            shuffled_train = train_idx[perm]

            for start in range(0, len(shuffled_train), batch_size):
                batch_ix = shuffled_train[start : start + batch_size]
                if len(batch_ix) == 0:
                    continue

                xb = torch.from_numpy(dataset.X[batch_ix]).to(device)
                ab = torch.from_numpy(dataset.actions[batch_ix]).to(device)
                adv_b = torch.from_numpy(advantages[batch_ix]).to(device)
                vt_b = torch.from_numpy(value_targets[batch_ix]).to(device)
                old_lp_b = old_log_probs_all[batch_ix].to(device)

                logits = actor(xb)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(ab)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv_b
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                values = critic(xb)
                value_loss = nn.functional.mse_loss(values, vt_b)

                loss = (
                    policy_loss
                    + value_loss_coef * value_loss
                    - entropy_coef * entropy
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()),
                    max_grad_norm,
                )
                optimizer.step()

                clipped = (ratio - 1.0).abs() > clip_epsilon
                epoch_stats["policy_loss"] += float(policy_loss.item())
                epoch_stats["value_loss"] += float(value_loss.item())
                epoch_stats["entropy"] += float(entropy.item())
                epoch_stats["clip_fraction"] += float(clipped.float().mean().item())
                epoch_stats["batches"] += 1

        n_batches = max(epoch_stats["batches"], 1)
        clip_frac = epoch_stats["clip_fraction"] / n_batches
        row = {
            "epoch": epoch,
            "policy_loss": epoch_stats["policy_loss"] / n_batches,
            "value_loss": epoch_stats["value_loss"] / n_batches,
            "entropy": epoch_stats["entropy"] / n_batches,
            "clip_fraction": clip_frac,
        }
        history.append(row)

        val_metric = _evaluate_val_return(critic, dataset, dataset.val_indices, device)
        row["val_value_mean"] = val_metric
        logger.info(
            "epoch %d/%d policy_loss=%.4f value_loss=%.4f entropy=%.4f "
            "clip_frac=%.1f%% val_value=%.3f",
            epoch,
            epochs,
            row["policy_loss"],
            row["value_loss"],
            row["entropy"],
            clip_frac * 100.0,
            val_metric,
        )
        if clip_frac > 0.5:
            logger.warning(
                "clip_fraction > 50%% - consider lowering learning rate (current %.2e)",
                lr,
            )

        if val_metric > best_val_metric:
            best_val_metric = val_metric

        epoch_entropy = row["entropy"]
        if epoch_entropy >= entropy_stop_threshold:
            _save_checkpoint(epoch_entropy, label="Saved")
        else:
            stop_epoch = epoch
            stop_reason = "entropy_below_threshold"
            if not saved_checkpoint:
                logger.warning(
                    "Entropy %.4f below %.2f before any healthy checkpoint - saving current weights",
                    epoch_entropy,
                    entropy_stop_threshold,
                )
                _save_checkpoint(epoch_entropy, label="Saved fallback")
            logger.warning(
                "Early stopping at epoch %d: entropy %.4f < %.2f (entropy collapse). "
                "Best saved entropy=%.4f",
                epoch,
                epoch_entropy,
                entropy_stop_threshold,
                best_entropy_at_save,
            )
            break

    if stop_reason is None and saved_checkpoint:
        stop_reason = "completed_epochs"
        stop_epoch = history[-1]["epoch"] if history else None

    # Reload best for config export
    if model_path.exists():
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["model_state_dict"])
        if "value_state_dict" in ckpt:
            critic.load_state_dict(ckpt["value_state_dict"])

    config = {
        "feature_layout": feature_layout(),
        "action_vocab": vocab_metadata(vocab),
        "algorithm": "ppo",
        "hyperparameters": {
            "learning_rate": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_epsilon": clip_epsilon,
            "value_loss_coef": value_loss_coef,
            "entropy_coef": entropy_coef,
            "entropy_stop": entropy_stop_threshold,
            "epochs": epochs,
            "ppo_epochs": ppo_epochs,
            "batch_size": batch_size,
            "terminal_reward_scale": terminal_reward_scale,
            "min_run_score_percentile": min_run_score_percentile,
            "val_fraction": val_fraction,
            "seed": seed,
        },
        "dataset_meta": dataset.meta,
        "history": history,
        "best_val_value_mean": best_val_metric,
        "best_entropy_at_save": best_entropy_at_save if saved_checkpoint else None,
        "entropy_stop_threshold": entropy_stop_threshold,
        "early_stop_epoch": stop_epoch,
        "early_stop_reason": stop_reason,
        "bc_init": str(start_from),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("Saved config to %s", config_path)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO fine-tune on decisions.jsonl")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--start-from",
        type=Path,
        default=DEFAULT_BC_MODEL,
        help="BC checkpoint to initialize actor (default: models/policy_net.pt)",
    )
    parser.add_argument("--model-out", type=Path, default=DEFAULT_PPO_MODEL)
    parser.add_argument("--config-out", type=Path, default=DEFAULT_PPO_CONFIG)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Inner PPO passes per batch")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--entropy-stop",
        type=float,
        default=0.8,
        help="Stop training when mean epoch entropy drops below this (default 0.8)",
    )
    parser.add_argument("--terminal-reward-scale", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--min-run-score-percentile", type=float, default=25.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_ppo(
        decisions_path=args.decisions,
        start_from=args.start_from,
        model_path=args.model_out,
        config_path=args.config_out,
        log_path=args.log,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        min_run_score_percentile=args.min_run_score_percentile,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device_name=args.device,
        terminal_reward_scale=args.terminal_reward_scale,
        max_grad_norm=args.max_grad_norm,
        entropy_stop_threshold=args.entropy_stop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
