"""PPO dataset reward shaping and GAE."""

from __future__ import annotations

import numpy as np

from training.ppo_dataset import (
    build_run_trajectories,
    compute_discounted_returns,
    compute_gae,
    normalize_rewards_by_return_std,
    scalar_immediate_reward,
)


def test_scalar_immediate_from_combat_dict() -> None:
    raw = {"combat_score_contribution": 2.5, "damage_dealt": 10}
    assert scalar_immediate_reward(raw) == 2.5


def test_discounted_returns_terminal_bonus() -> None:
    rewards = np.array([0.0, 0.0, 10.0], dtype=np.float32)
    returns = compute_discounted_returns(rewards, gamma=0.99)
    assert returns[2] == 10.0
    assert returns[0] > 0.0


def test_gae_episode_end() -> None:
    rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5, 0.5, 0.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    adv, targets = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    assert len(adv) == 3
    assert len(targets) == 3


def test_normalize_rewards_by_return_std() -> None:
    step = np.array([1.0, 2.0, 100.0], dtype=np.float32)
    ret = np.array([10.0, 50.0, 200.0], dtype=np.float32)
    step_n, ret_n, scale = normalize_rewards_by_return_std(step, ret)
    assert scale == ret.std()
    assert abs(ret_n.std() - 1.0) < 0.05
    assert np.allclose(step_n, step / scale)


def test_build_run_trajectories_groups_by_run() -> None:
    rows = [
        {
            "run_id": "r1",
            "timestamp": "t1",
            "immediate_reward": 1.0,
            "run_outcome": {"reward": 100.0},
            "action_taken": {"action": "end_turn"},
            "state_snapshot": {},
        },
        {
            "run_id": "r1",
            "timestamp": "t2",
            "immediate_reward": 2.0,
            "run_outcome": {"reward": 100.0},
            "action_taken": {"action": "end_turn"},
            "state_snapshot": {},
        },
    ]
    trajs = build_run_trajectories(rows, {"r1": 100.0}, gamma=0.99)
    assert len(trajs) == 1
    assert len(trajs[0].rows) == 2
    assert trajs[0].rewards[-1] == 2.0 + 100.0
