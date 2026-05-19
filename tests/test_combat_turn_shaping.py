"""Combat per-step reward shaping."""

import pytest

from sts2_agent.scorer import combat_turn_shaping


def test_combat_turn_shaping_aggressive_trade():
    # 3 HP lost but 12 damage dealt: net positive under new weights
    assert combat_turn_shaping(3, 0, 12) == pytest.approx(10.5)


def test_combat_turn_shaping_block_and_damage():
    assert combat_turn_shaping(0, 5, 8) == pytest.approx(9.5)


def test_combat_turn_shaping_dynamic_multipliers():
    # aggressive: more damage reward, less HP penalty
    assert combat_turn_shaping(4, 0, 10, damage_mult=2.0, hp_loss_mult=0.25) == pytest.approx(19.0)
