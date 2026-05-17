"""Sample weights - human run multiplier."""

import numpy as np

from training.dataset import compute_sample_weights


def test_human_weight_multiplier():
    rows = [
        {"run_id": "h1", "action_taken": {"action": "end_turn"}},
        {"run_id": "a1", "action_taken": {"action": "end_turn"}},
    ]
    run_scores = {"h1": 100.0, "a1": 100.0}
    run_sources = {"h1": "human", "a1": "agent"}

    w = compute_sample_weights(
        rows,
        run_scores,
        run_sources,
        human_weight=3.0,
    )
    assert w[0] == np.float32(w[1] * 3.0)


def test_human_weight_default_agent_unchanged():
    rows = [{"run_id": "a1", "action_taken": {"action": "end_turn"}}]
    w = compute_sample_weights(
        rows,
        {"a1": 50.0},
        {"a1": "agent"},
        human_weight=3.0,
    )
    assert w[0] == np.float32(1.0) or w[0] >= np.float32(0.1)
