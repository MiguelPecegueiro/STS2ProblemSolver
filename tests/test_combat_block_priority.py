"""Block vs attack priority when enemies are not attacking."""

from __future__ import annotations

from sts2_agent.knowledge import get_knowledge
from sts2_agent.scorer import (
    card_applies_vulnerable,
    intent_is_attack,
    score_combat_play,
    total_incoming_attack_damage,
)


def test_intent_is_attack_rejects_defend_and_buff() -> None:
    assert not intent_is_attack({"type": "defend", "damage": 0})
    assert not intent_is_attack({"type": "buff", "text": "Gain 2 Strength"})
    assert not intent_is_attack({"type": "unknown"})
    assert intent_is_attack({"type": "attack", "damage": 12})


def test_total_incoming_zero_on_defend_turn() -> None:
    enemies = [
        {
            "name": "Slime",
            "hp": 20,
            "intent": {"type": "defend", "text": "Gain 7 Block"},
        }
    ]
    total, reasons = total_incoming_attack_damage(enemies)
    assert total == 0
    assert any("not counted" in r for r in reasons)


def test_empower_intent_not_counted_as_incoming() -> None:
    enemies = [
        {
            "name": "Fuzzy Wurm Crawler",
            "hp": 20,
            "intent": {"title": "Empower", "type": "buff", "damage": 0},
        }
    ]
    total, reasons = total_incoming_attack_damage(enemies)
    assert total == 0
    assert not intent_is_attack(enemies[0]["intent"])


def test_bash_scores_above_strike_when_enemy_not_vulnerable() -> None:
    kb = get_knowledge()
    bash = {"name": "Bash", "cost": 2, "damage": 8, "type": "attack"}
    strike = {"name": "Strike", "cost": 1, "damage": 6, "type": "attack"}
    enemies = [{"hp": 30, "name": "Fuzzy Wurm Crawler"}]
    assert card_applies_vulnerable(bash, kb)
    bash_score = score_combat_play(
        bash, energy=3, incoming_damage=0, current_block=0, enemies=enemies, kb=kb, floor=1
    )
    strike_score = score_combat_play(
        strike,
        energy=3,
        incoming_damage=0,
        current_block=0,
        enemies=enemies,
        kb=kb,
        floor=1,
    )
    assert bash_score.score > strike_score.score


def test_block_scores_below_attack_when_no_incoming() -> None:
    kb = get_knowledge()
    strike = {"name": "Strike", "cost": 1, "damage": 6, "type": "attack"}
    defend = {"name": "Defend", "cost": 1, "block": 5, "type": "skill"}
    atk = score_combat_play(
        strike,
        energy=3,
        incoming_damage=0,
        current_block=0,
        enemies=[{"hp": 30, "name": "Foo"}],
        kb=kb,
        floor=5,
    )
    blk = score_combat_play(
        defend,
        energy=3,
        incoming_damage=0,
        current_block=0,
        enemies=[{"hp": 30, "name": "Foo"}],
        kb=kb,
        floor=5,
    )
    assert atk.score > blk.score
