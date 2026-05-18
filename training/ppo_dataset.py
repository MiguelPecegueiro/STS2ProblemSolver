"""Load decisions.jsonl as temporally ordered trajectories for offline PPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.actions import action_to_key, build_action_vocab, encode_action
from training.dataset import (
    DEFAULT_DECISIONS_PATH,
    DEFAULT_RUNS_PATH,
    load_decision_rows,
    load_run_scores,
)
from training.features import FEATURE_DIM, encode_snapshot

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, dict):
            if isinstance(value, dict):
                for key in (
                    "combat_score_contribution",
                    "reward",
                    "value",
                    "score",
                ):
                    if value.get(key) is not None:
                        return float(value[key])
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def scalar_immediate_reward(raw: Any) -> float:
    """Normalize immediate_reward field to a scalar step reward."""
    return _safe_float(raw, 0.0)


def scalar_run_outcome_reward(row: dict, run_scores: dict[str, float]) -> float:
    outcome = row.get("run_outcome") or {}
    if outcome.get("reward") is not None:
        return float(outcome["reward"])
    if outcome.get("run_score") is not None:
        return float(outcome["run_score"])
    rid = str(row.get("run_id") or "")
    if rid in run_scores:
        return float(run_scores[rid])
    return 0.0


@dataclass
class RunTrajectory:
    run_id: str
    rows: list[dict]
    rewards: np.ndarray
    returns: np.ndarray
    indices: list[int]


@dataclass
class PPODataset:
    """Flat transition store with run-boundary metadata for GAE."""

    X: np.ndarray
    actions: np.ndarray
    step_rewards: np.ndarray
    returns: np.ndarray
    run_ids: list[str]
    run_starts: np.ndarray
    run_lengths: np.ndarray
    val_indices: np.ndarray
    train_indices: np.ndarray
    vocab: dict[str, int]
    meta: dict[str, Any]

    def __len__(self) -> int:
        return len(self.actions)


def normalize_rewards_by_return_std(
    step_rewards: np.ndarray,
    returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale step rewards and returns by dataset return std (stable critic targets)."""
    scale = float(np.std(returns))
    if scale < 1e-8:
        scale = 1.0
    return (
        (step_rewards / scale).astype(np.float32),
        (returns / scale).astype(np.float32),
        scale,
    )


def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Monte Carlo returns G_t = sum_{k>=0} gamma^k r_{t+k}."""
    out = np.zeros_like(rewards, dtype=np.float32)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = float(rewards[t]) + gamma * g
        out[t] = g
    return out


def build_run_trajectories(
    rows: list[dict],
    run_scores: dict[str, float],
    *,
    gamma: float = 0.99,
    terminal_reward_scale: float = 1.0,
) -> list[RunTrajectory]:
    """Group rows by run_id (temporal order preserved) and assign step rewards."""
    by_run: dict[str, list[dict]] = {}
    for row in rows:
        rid = str(row.get("run_id") or "")
        if not rid:
            continue
        by_run.setdefault(rid, []).append(row)

    trajectories: list[RunTrajectory] = []
    for rid, run_rows in by_run.items():
        if not run_rows:
            continue
        run_rows.sort(key=lambda r: str(r.get("timestamp") or ""))
        n = len(run_rows)
        rewards = np.zeros(n, dtype=np.float32)
        for i, row in enumerate(run_rows):
            rewards[i] = scalar_immediate_reward(row.get("immediate_reward"))
        terminal = scalar_run_outcome_reward(run_rows[-1], run_scores) * terminal_reward_scale
        rewards[-1] += terminal
        returns = compute_discounted_returns(rewards, gamma)
        trajectories.append(
            RunTrajectory(
                run_id=rid,
                rows=run_rows,
                rewards=rewards,
                returns=returns,
                indices=list(range(n)),
            )
        )
    return trajectories


def trajectories_to_arrays(
    trajectories: list[RunTrajectory],
    vocab: dict[str, int],
    *,
    unknown_action_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    actions: list[int] = []
    step_rewards: list[float] = []
    returns: list[float] = []
    run_ids: list[str] = []
    run_starts: list[int] = []
    run_lengths: list[int] = []

    offset = 0
    for traj in trajectories:
        run_starts.append(offset)
        count = 0
        for i, row in enumerate(traj.rows):
            label = encode_action(
                row.get("action_taken"),
                vocab,
                snapshot=row.get("state_snapshot"),
                unknown_index=unknown_action_id,
            )
            if label is None:
                continue
            xs.append(
                encode_snapshot(
                    row.get("state_snapshot"),
                    state_type=str(row.get("state_type") or ""),
                    floor=int(row.get("floor") or 0),
                    act=int(row.get("act") or 1),
                    immediate_reward=scalar_immediate_reward(row.get("immediate_reward")),
                )
            )
            actions.append(label)
            step_rewards.append(float(traj.rewards[i]))
            returns.append(float(traj.returns[i]))
            run_ids.append(traj.run_id)
            count += 1
        run_lengths.append(count)
        offset += count

    if not xs:
        empty = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        return (
            empty,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    return (
        np.stack(xs, axis=0),
        np.array(actions, dtype=np.int64),
        np.array(step_rewards, dtype=np.float32),
        np.array(returns, dtype=np.float32),
        run_ids,
        np.array(run_starts, dtype=np.int64),
        np.array(run_lengths, dtype=np.int64),
    )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
  Compute GAE advantages and value targets for one trajectory segment.

    rewards, values, dones: shape (T,)
    """
    t_len = len(rewards)
    advantages = np.zeros(t_len, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(t_len)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = float(values[t + 1]) if t + 1 < len(values) else 0.0
        delta = (
            float(rewards[t])
            + gamma * next_value * next_non_terminal
            - float(values[t])
        )
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    targets = advantages + values[:t_len]
    return advantages, targets.astype(np.float32)


def compute_gae_for_dataset(
    step_rewards: np.ndarray,
    values: np.ndarray,
    run_starts: np.ndarray,
    run_lengths: np.ndarray,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE per run; values shape (N,) aligned with flat transitions."""
    n = len(step_rewards)
    advantages = np.zeros(n, dtype=np.float32)
    targets = np.zeros(n, dtype=np.float32)
    for start, length in zip(run_starts, run_lengths):
        end = int(start) + int(length)
        if end > n:
            continue
        r = step_rewards[start:end]
        v = values[start:end]
        # Append bootstrap value 0 at terminal (episode end)
        v_pad = np.append(v, 0.0)
        dones = np.zeros(len(r), dtype=np.float32)
        dones[-1] = 1.0
        adv, tgt = compute_gae(r, v_pad, dones, gamma=gamma, gae_lambda=gae_lambda)
        advantages[start:end] = adv
        targets[start:end] = tgt
    return advantages, targets


def load_vocab_from_config(config_path: Path) -> dict[str, int]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    vocab = data.get("action_vocab", {}).get("action_to_id")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError(f"No action_to_id in {config_path}")
    return {str(k): int(v) for k, v in vocab.items()}


def build_ppo_dataset(
    decisions_path: Path = DEFAULT_DECISIONS_PATH,
    *,
    runs_path: Path | None = None,
    gamma: float = 0.99,
    min_run_score_percentile: float = 25.0,
    min_game_version: str | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
    terminal_reward_scale: float = 1.0,
    vocab: dict[str, int] | None = None,
    clean_only: bool = True,
) -> PPODataset:
    rows, run_scores, filter_meta = load_decision_rows(
        decisions_path,
        runs_path=runs_path or DEFAULT_RUNS_PATH,
        min_run_score_percentile=min_run_score_percentile,
        min_game_version=min_game_version,
        clean_only=clean_only,
    )
    if not rows:
        raise ValueError(f"No PPO rows after filtering: {decisions_path}")

    if vocab is None:
        vocab = build_action_vocab(decisions_path)
    trajectories = build_run_trajectories(
        rows,
        run_scores,
        gamma=gamma,
        terminal_reward_scale=terminal_reward_scale,
    )
    if not trajectories:
        raise ValueError("No trajectories built from decisions")

    X, actions, step_rewards, returns, run_ids, run_starts, run_lengths = (
        trajectories_to_arrays(trajectories, vocab)
    )

    raw_reward_stats = {
        "step_reward_mean": float(step_rewards.mean()),
        "step_reward_std": float(step_rewards.std() + 1e-8),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std() + 1e-8),
    }
    step_rewards, returns, reward_scale = normalize_rewards_by_return_std(
        step_rewards, returns
    )

    # Train/val split by run_id
    unique_runs = sorted({t.run_id for t in trajectories})
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_runs)
    n_val = max(1, int(len(unique_runs) * val_fraction))
    val_run_set = set(unique_runs[:n_val])

    all_idx = np.arange(len(actions), dtype=np.int64)
    val_mask = np.array([rid in val_run_set for rid in run_ids], dtype=bool)
    val_indices = all_idx[val_mask]
    train_indices = all_idx[~val_mask]

    reward_stats = {
        "raw": raw_reward_stats,
        "reward_scale": reward_scale,
        "step_reward_mean": float(step_rewards.mean()),
        "step_reward_std": float(step_rewards.std() + 1e-8),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std() + 1e-8),
    }

    meta = {
        "num_transitions": int(len(actions)),
        "num_runs": len(trajectories),
        "num_train": int(len(train_indices)),
        "num_val": int(len(val_indices)),
        "gamma": gamma,
        "reward_stats": reward_stats,
        "reward_scale": reward_scale,
        "terminal_reward_scale": terminal_reward_scale,
        "clean_only": clean_only,
        "filter": filter_meta,
    }

    return PPODataset(
        X=X,
        actions=actions,
        step_rewards=step_rewards,
        returns=returns,
        run_ids=run_ids,
        run_starts=run_starts,
        run_lengths=run_lengths,
        val_indices=val_indices,
        train_indices=train_indices,
        vocab=vocab,
        meta=meta,
    )
