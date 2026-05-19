"""Combat solver: sequence search, lethal detection, caching."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sts2_agent.combat_solver import (
    SolverContext,
    clear_turn_plan_cache,
    set_combat_solver_enabled,
    try_solver_decide,
    _enumerate_sequences,
    _initial_enemy_hp,
    _pick_best_sequence,
    _player_outgoing_multiplier,
    _simulate_play,
    _SimState,
)
from sts2_agent.knowledge import KnowledgeBase


class _StubKB(KnowledgeBase):
    def __init__(self) -> None:
        self._cards = {
            "strike": {"type_key": "attack", "damage": 6, "cost": 1},
            "bash": {"type_key": "attack", "damage": 8, "cost": 2},
            "defend": {"type_key": "skill", "block": 5, "cost": 1},
            "demon form": {"type_key": "power", "cost": 3, "powers_applied": [{"power": "strength", "amount": 2}]},
        }

    def lookup_card(self, name: str):
        return self._cards.get(str(name).lower(), {})


def _combat_state(
    *,
    hand: list[dict],
    energy: int = 3,
    enemy_hp: int = 10,
    incoming: int = 0,
    player_hp: int = 50,
    player_block: int = 0,
    draw_pile: list | None = None,
) -> dict:
    return {
        "state_type": "monster",
        "run": {"id": "test-run", "floor": 5},
        "battle": {
            "turn": "player",
            "round": 1,
            "is_play_phase": True,
            "enemies": [
                {
                    "name": "Slime",
                    "hp": enemy_hp,
                    "max_hp": enemy_hp,
                    "entity_id": "slime_0",
                    "intent": {"type": "attack", "damage": incoming},
                }
            ],
        },
        "player": {
            "hp": player_hp,
            "max_hp": 80,
            "energy": energy,
            "block": player_block,
            "hand": hand,
            "draw_pile": draw_pile or [],
            "discard_pile": [],
        },
    }


@pytest.fixture(autouse=True)
def _reset_solver() -> None:
    clear_turn_plan_cache()
    set_combat_solver_enabled(True)
    yield
    clear_turn_plan_cache()
    set_combat_solver_enabled(True)


def test_lethal_two_strikes() -> None:
    kb = _StubKB()
    hand = [
        {"index": 0, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
        {"index": 1, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
    ]
    state = _combat_state(hand=hand, energy=2, enemy_hp=10, incoming=0)
    ctx = SolverContext(
        state=state,
        kb=kb,
        hand=hand,
        energy=2,
        player_block=0,
        player_hp=50,
        incoming=0,
        next_incoming=0,
        living=state["battle"]["enemies"],
    )
    outcomes = _enumerate_sequences(ctx)
    best, tag, _ = _pick_best_sequence(outcomes, ctx)
    assert tag == "solver: lethal T1"
    assert best.is_lethal
    assert best.total_damage >= 10


def test_block_reduces_hp_lost_in_trade() -> None:
    kb = _StubKB()
    hand = [
        {"index": 0, "id": "DEFEND", "name": "Defend", "cost": 1, "type": "skill", "can_play": True},
        {"index": 1, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
    ]
    state = _combat_state(hand=hand, energy=2, enemy_hp=30, incoming=8, player_hp=40)
    ctx = SolverContext(
        state=state,
        kb=kb,
        hand=hand,
        energy=2,
        player_block=0,
        player_hp=40,
        incoming=8,
        next_incoming=0,
        living=state["battle"]["enemies"],
    )
    outcomes = _enumerate_sequences(ctx)
    best, tag, _ = _pick_best_sequence(outcomes, ctx)
    assert tag == "solver: trade"
    assert best.total_block >= 5
    assert best.hp_lost <= 3


def test_weak_reduces_outgoing_damage() -> None:
    player = {"status": [{"id": "WEAK", "amount": 1}]}
    assert _player_outgoing_multiplier(player) == 0.75


def test_vulnerable_multiplies_damage() -> None:
    kb = _StubKB()
    hand = [
        {"index": 0, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
    ]
    sim = _SimState(
        energy=3,
        enemy_hp={"slime_0": 20},
        enemy_vuln={"slime_0"},
        block_gained=0,
        strength=0,
        outgoing_mult=1.0,
        used=frozenset(),
        steps=[],
    )
    nxt = _simulate_play(sim, hand, kb, 0, "slime_0")
    assert nxt is not None
    assert nxt.enemy_hp["slime_0"] == 11  # 6 * 1.5 = 9 damage


def test_empty_hand_end_turn() -> None:
    state = _combat_state(hand=[], energy=3, enemy_hp=10)
    action, reasons = try_solver_decide(state)
    assert action == {"action": "end_turn"}
    assert any("empty hand" in r for r in reasons)


@patch("sts2_agent.combat_solver._solve_turn", side_effect=RuntimeError("boom"))
def test_solver_error_returns_none_for_legacy(_mock: object) -> None:
    state = _combat_state(
        hand=[
            {"index": 0, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
        ],
        energy=3,
    )
    result = try_solver_decide(state)
    assert result is None


def test_turn_plan_cache_executes_second_step() -> None:
    kb = _StubKB()
    hand = [
        {"index": 0, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
        {"index": 1, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
    ]
    state = _combat_state(hand=hand, energy=2, enemy_hp=20, incoming=0)

    with patch("sts2_agent.combat_solver.get_knowledge", return_value=kb):
        with patch("sts2_agent.combat_solver.total_incoming_attack_damage", return_value=(0, [])):
            with patch("sts2_agent.combat_solver.next_turn_combat_estimates") as mock_est:
                mock_est.return_value.expected_damage = 0
                mock_est.return_value.expected_block = 0
                mock_est.return_value.reasons = []
                a1, r1 = try_solver_decide(state)
                assert a1 is not None
                assert a1.get("action") == "play_card"

                state2 = dict(state)
                state2["player"] = dict(state["player"])
                state2["player"]["hand"] = [hand[1]]
                a2, r2 = try_solver_decide(state2)
                assert any("cached plan" in r for r in r2)


def test_max_depth_limits_sequences() -> None:
    kb = _StubKB()
    hand = [
        {
            "index": i,
            "id": "STRIKE",
            "name": "Strike",
            "cost": 1,
            "type": "attack",
            "can_play": True,
        }
        for i in range(10)
    ]
    state = _combat_state(hand=hand, energy=10, enemy_hp=100)
    ctx = SolverContext(
        state=state,
        kb=kb,
        hand=hand,
        energy=10,
        player_block=0,
        player_hp=50,
        incoming=0,
        next_incoming=0,
        living=state["battle"]["enemies"],
    )
    outcomes = _enumerate_sequences(ctx)
    assert outcomes
    assert max(len(o.steps) for o in outcomes) <= 8


def test_initial_enemy_hp_map() -> None:
    living = [{"entity_id": "a", "hp": 5}, {"entity_id": "b", "hp": 0}]
    assert _initial_enemy_hp(living) == {"a": 5}


def test_kill_attacker_beats_pointless_block_multi_enemy() -> None:
    """6-dmg attacker (4 incoming) + passive enemy: strike first, no Defend."""
    kb = _StubKB()
    hand = [
        {"index": 0, "id": "DEFEND", "name": "Defend", "cost": 1, "type": "skill", "can_play": True},
        {"index": 1, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
    ]
    living = [
        {
            "name": "Slime",
            "hp": 6,
            "entity_id": "attacker",
            "intent": {"type": "attack", "damage": 4},
        },
        {
            "name": "Idle",
            "hp": 20,
            "entity_id": "idle",
            "intent": {"type": "defend"},
        },
    ]
    state = {
        "state_type": "monster",
        "run": {"id": "test-run", "floor": 5},
        "battle": {"turn": "player", "round": 1, "enemies": living},
        "player": {"hp": 50, "max_hp": 80, "energy": 2, "block": 0, "hand": hand},
    }
    ctx = SolverContext(
        state=state,
        kb=kb,
        hand=hand,
        energy=2,
        player_block=0,
        player_hp=50,
        incoming=4,
        next_incoming=0,
        living=living,
    )
    outcomes = _enumerate_sequences(ctx)
    best, tag, _ = _pick_best_sequence(outcomes, ctx)
    assert tag in ("solver: kill removes incoming", "solver: lethal T1", "solver: trade")
    assert best.hp_lost == 0
    assert best.steps[0].card_label == "Strike"
    assert "attacker" in (best.steps[0].target_entity_id or "")


def test_no_incoming_does_not_favor_block() -> None:
    kb = _StubKB()
    hand = [
        {"index": 0, "id": "DEFEND", "name": "Defend", "cost": 1, "type": "skill", "can_play": True},
        {"index": 1, "id": "STRIKE", "name": "Strike", "cost": 1, "type": "attack", "can_play": True},
    ]
    state = _combat_state(hand=hand, energy=2, enemy_hp=30, incoming=0)
    ctx = SolverContext(
        state=state,
        kb=kb,
        hand=hand,
        energy=2,
        player_block=0,
        player_hp=50,
        incoming=0,
        next_incoming=0,
        living=state["battle"]["enemies"],
    )
    outcomes = _enumerate_sequences(ctx)
    best, _, _ = _pick_best_sequence(outcomes, ctx)
    assert best.steps[0].card_label == "Strike"
    assert best.total_block == 0
