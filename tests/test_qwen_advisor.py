"""Tests for Qwen combat strategy advisor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from sts2_agent.qwen_advisor import (
    DEFAULT_DAMAGE_MULT,
    DEFAULT_HP_LOSS_MULT,
    QwenAdvisor,
    _enemy_name_lookup_keys,
    _format_enemy_section,
    build_combat_prompt,
    configure_qwen,
    lookup_enemy_knowledge,
    parse_strategy_response,
    starter_deck_ids_for_state,
)
from sts2_agent.state_parse import is_player_combat_turn


def test_parse_strategy_response_valid_json():
    text = json.dumps(
        {
            "strategy": "aggressive",
            "damage_multiplier": 1.8,
            "hp_loss_multiplier": 0.3,
            "reasoning": "Kill scaling enemy quickly.",
        }
    )
    result = parse_strategy_response(text)
    assert result is not None
    assert result.strategy == "aggressive"
    assert result.damage_mult == pytest.approx(1.8)
    assert result.hp_loss_mult == pytest.approx(0.3)
    assert result.source == "qwen"
    assert "scaling" in result.reasoning


def test_parse_strategy_response_markdown_fence():
    text = '```json\n{"strategy": "defensive", "damage_multiplier": 0.6, "hp_loss_multiplier": 0.9, "reasoning": "Survive."}\n```'
    result = parse_strategy_response(text)
    assert result is not None
    assert result.strategy == "defensive"
    assert result.damage_mult == pytest.approx(0.6)


def test_parse_strategy_response_clamps_multipliers():
    text = json.dumps(
        {
            "strategy": "balanced",
            "damage_multiplier": 9.0,
            "hp_loss_multiplier": 0.01,
            "reasoning": "x",
        }
    )
    result = parse_strategy_response(text)
    assert result is not None
    assert result.damage_mult == 2.0
    assert result.hp_loss_mult == 0.25


def test_parse_strategy_response_invalid_returns_none():
    assert parse_strategy_response("not json at all") is None


def test_build_combat_prompt_uses_draw_pile_when_deck_empty():
    state = {
        "run": {"floor": 1, "act": 1},
        "player": {
            "hp": 50,
            "max_hp": 70,
            "deck": [],
            "draw_pile": [{"name": "Strike"}, {"name": "Defend"}],
            "hand": [],
        },
    }
    prompt = build_combat_prompt(
        state,
        combat_type="monster",
        enemy_names=["Slime"],
        expert_enemies={},
    )
    assert "Strike" in prompt
    assert "Defend" in prompt


def test_build_combat_prompt_uses_deck_card_ids_fallback():
    prompt = build_combat_prompt(
        {"run": {"floor": 1}, "player": {"hp": 50, "max_hp": 70, "deck": []}},
        combat_type="monster",
        enemy_names=["Slime"],
        expert_enemies={},
        deck_card_ids=["STRIKE", "DEFEND"],
    )
    assert "STRIKE" in prompt or "Strike" in prompt


def test_build_combat_prompt_includes_state():
    state = {
        "run": {"floor": 12, "act": 2},
        "player": {
            "hp": 45,
            "max_hp": 70,
            "energy": 3,
            "deck": [{"name": "Strike"}, {"name": "Defend"}],
            "hand": [],
            "draw_pile": [{"name": "Bash"}],
        },
    }
    prompt = build_combat_prompt(
        state,
        combat_type="elite",
        enemy_names=["Phrog Parasite"],
        expert_enemies={"phrog parasite": {"notes": "Spawns minions"}},
    )
    assert "elite" in prompt
    assert "Phrog Parasite" in prompt
    assert "Spawns minions" in prompt
    assert "Strike" in prompt
    assert "Cards will be drawn at turn start" in prompt
    assert "Deck composition is:" in prompt
    assert "Hand:" not in prompt
    assert "Floor: 12" in prompt


def test_starter_deck_for_ironclad_floor_one():
    state = {
        "run": {"floor": 1, "character": "IRONCLAD"},
        "player": {"hp": 70, "max_hp": 70, "deck": [], "hand": []},
    }
    ids = starter_deck_ids_for_state(state)
    assert len(ids) == 10
    assert ids.count("BASH") == 1


def test_is_player_combat_turn_rejects_empty_turn_label():
    state = {
        "state_type": "monster",
        "battle": {"is_play_phase": True, "turn": ""},
        "player": {"hand": []},
    }
    assert not is_player_combat_turn(state)
    state["battle"]["turn"] = "player"
    assert is_player_combat_turn(state)


def test_build_combat_prompt_unknown_enemy_prefers_aggression():
    prompt = build_combat_prompt(
        {"run": {"floor": 1, "act": 1}, "player": {"hp": 50, "max_hp": 70, "deck": []}},
        combat_type="monster",
        enemy_names=["Mystery Beast"],
        expert_enemies={},
    )
    assert "prefer aggression as default" in prompt
    assert "unknown mechanics" in prompt
    assert "[missing]" in prompt


def test_enemy_lookup_strips_size_suffix():
    expert = {"twig slime": {"notes": "Splits on death"}}
    entry, source = lookup_enemy_knowledge("Twig Slime (S)", expert)
    assert source == "expert_knowledge"
    assert entry is not None


def test_enemy_lookup_uses_learned_compendium(monkeypatch):
    monkeypatch.setattr(
        "sts2_agent.qwen_advisor.lookup_learned_compendium",
        lambda _name: {
            "fight_count": 5,
            "learned_cycle": ["aggressive|d4|b0"],
            "moves": {
                "aggressive|d4|b0": {
                    "damage": 4,
                    "tags": ["attack"],
                    "seen_count": 10,
                }
            },
        },
    )
    entry, source = lookup_enemy_knowledge("Twig Slime (S)", {})
    assert source == "learned_compendium"
    assert "intent cycle" in entry.get("notes", "")
    assert "Aggressive" in entry.get("notes", "")


def test_format_enemy_section_shows_source_tag():
    text = _format_enemy_section(
        "Kin Follower",
        {"notes": "Spawns allies"},
        "expert_knowledge",
    )
    assert "[expert_knowledge]" in text
    assert "Spawns allies" in text


@patch("sts2_agent.qwen_advisor.requests.post")
def test_full_prompt_logged_at_debug(mock_post, caplog):
    import logging

    configure_qwen(enabled=True, combat_enabled=True, log_full_prompt=True)
    caplog.set_level(logging.DEBUG, logger="sts2_agent.qwen_advisor")
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "choices": [{"message": {"content": '{"strategy":"balanced","damage_multiplier":1.0,"hp_loss_multiplier":0.5,"reasoning":"ok"}'}}]
        },
    )
    state = {
        "run": {"floor": 1, "act": 1, "character": "IRONCLAD"},
        "player": {"hp": 50, "max_hp": 70, "hand": [{"name": "Strike"}], "deck": []},
        "battle": {"turn": "player", "is_play_phase": True},
    }
    QwenAdvisor().begin_fight(state, combat_type="monster", enemy_names=["Slime"])
    assert "======== USER ========" in caplog.text
    assert "Combat type: monster" in caplog.text


def test_build_combat_prompt_knowledge_debug_log(caplog):
    import logging

    caplog.set_level(logging.DEBUG, logger="sts2_agent.qwen_advisor")
    build_combat_prompt(
        {"run": {"floor": 1}, "player": {"hp": 1, "max_hp": 1, "deck": []}},
        combat_type="elite",
        enemy_names=["Known", "Unknown"],
        expert_enemies={"known": {"notes": "test"}},
    )
    assert "Qwen knowledge injection: expert=" in caplog.text
    assert "Known" in caplog.text
    assert "Unknown" in caplog.text


@patch("sts2_agent.qwen_advisor.requests.post")
def test_fetch_strategy_applies_multipliers(mock_post):
    configure_qwen(enabled=True, combat_enabled=True, timeout=5.0)
    mock_post.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "strategy": "aggressive",
                                "damage_multiplier": 1.5,
                                "hp_loss_multiplier": 0.4,
                                "reasoning": "Burst down effigy.",
                            }
                        )
                    }
                }
            ]
        },
    )
    advisor = QwenAdvisor()
    state = {
        "run": {"floor": 8, "act": 1},
        "player": {"hp": 50, "max_hp": 70, "energy": 3, "deck": [], "hand": []},
        "battle": {"enemies": [{"name": "Bygone Effigy", "hp": 40}]},
    }
    mult = advisor.begin_fight(state, combat_type="elite", enemy_names=["Bygone Effigy"])
    assert mult.source == "qwen"
    assert mult.damage_mult == pytest.approx(1.5)
    assert mult.hp_loss_mult == pytest.approx(0.4)
    record = advisor.end_fight()
    assert record is not None
    assert record["strategy"] == "aggressive"


@patch("sts2_agent.qwen_advisor.requests.post")
def test_connection_error_uses_defaults(mock_post):
    configure_qwen(enabled=True)
    mock_post.side_effect = requests.ConnectionError("refused")
    advisor = QwenAdvisor()
    state = {
        "run": {"floor": 1, "act": 1},
        "player": {"hp": 70, "max_hp": 70, "energy": 3, "deck": [], "hand": []},
    }
    mult = advisor.begin_fight(state, combat_type="monster", enemy_names=["Slime"])
    assert mult.damage_mult == DEFAULT_DAMAGE_MULT
    assert mult.hp_loss_mult == DEFAULT_HP_LOSS_MULT
    assert mult.source == "default"


@patch("sts2_agent.qwen_advisor.requests.post")
def test_timeout_uses_defaults(mock_post):
    configure_qwen(enabled=True, combat_enabled=True)
    mock_post.side_effect = requests.Timeout("slow")
    advisor = QwenAdvisor()
    state = {
        "run": {"floor": 1, "act": 1},
        "player": {"hp": 70, "max_hp": 70, "energy": 3, "deck": [], "hand": []},
    }
    mult = advisor.begin_fight(state, combat_type="monster", enemy_names=["Slime"])
    assert mult.source == "timeout"


def test_disabled_skips_api_call():
    configure_qwen(enabled=False)
    advisor = QwenAdvisor()
    with patch("sts2_agent.qwen_advisor.requests.post") as mock_post:
        mult = advisor.begin_fight(
            {"run": {"floor": 1}, "player": {"hp": 1, "max_hp": 1}},
            combat_type="monster",
            enemy_names=["X"],
        )
        mock_post.assert_not_called()
    assert mult.source == "default"


def test_combat_disabled_skips_api_call():
    configure_qwen(enabled=True, combat_enabled=False, macro_enabled=True)
    advisor = QwenAdvisor()
    with patch("sts2_agent.qwen_advisor.requests.post") as mock_post:
        mult = advisor.begin_fight(
            {"run": {"floor": 1}, "player": {"hp": 1, "max_hp": 1}},
            combat_type="monster",
            enemy_names=["X"],
        )
        mock_post.assert_not_called()
    assert mult.source == "default"
