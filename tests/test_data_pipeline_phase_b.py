"""Phase B logging: card_reward_offered, combat_summary, killing_enemy."""

from __future__ import annotations

from unittest.mock import patch

from sts2_agent.data_pipeline import DataPipeline, _enemy_names_from_state
from sts2_agent.state_parse import extract_card_reward_offered


def test_extract_card_reward_offered():
    state = {
        "card_reward": {
            "cards": [
                {"name": "Strike", "id": "strike"},
                {"name": "Defend", "id": "defend"},
                {"name": "Bash", "id": "bash"},
            ]
        }
    }
    assert extract_card_reward_offered(state) == ["Strike", "Defend", "Bash"]

    # Name-only payloads (no id) must still log all offers
    name_only = {
        "card_reward": {"cards": [{"name": "Strike"}, {"name": "Defend"}, {"name": "Bash"}]}
    }
    assert extract_card_reward_offered(name_only) == ["Strike", "Defend", "Bash"]


def test_enemy_names_from_state_dedupes():
    state = {
        "battle": {
            "enemies": [
                {"name": "Jaw Worm", "entity_id": "a"},
                {"name": "Jaw Worm", "entity_id": "b"},
                {"name": "Slime", "entity_id": "c"},
            ]
        }
    }
    assert _enemy_names_from_state(state) == ["Jaw Worm", "Slime"]


def test_combat_summary_appended_on_end_combat():
    pipe = DataPipeline()
    pipe._run_active = True
    pipe.run_id = "test-run"
    pipe._in_combat = True
    pipe._combat_state_type = "elite"
    pipe._combat_start_hp = 50
    pipe._combat_start_max_hp = 50
    pipe._combat_enemy_names = ["Jaw Worm"]
    pipe._combat_decision_indices = [0, 1, 2]
    pipe._combat_damage_dealt_fight = 42
    pipe._enemy_hp_snapshot = {}

    state = {
        "player": {"hp": 30, "max_hp": 50},
        "battle": {"enemies": []},
    }
    with patch("sts2_agent.enemy_compendium.finalize_combat_observation"), patch(
        "sts2_agent.potions.clear_potion_session_failures"
    ), patch("sts2_agent.potions.get_potion_drop_tracker") as tracker:
        tracker.return_value.note_combat_ended = lambda *_: None
        pipe._end_combat(state)

    assert len(pipe._combat_summaries) == 1
    summary = pipe._combat_summaries[0]
    assert summary["enemy_names"] == ["Jaw Worm"]
    assert summary["turns"] == 3
    assert summary["damage_taken"] == 20
    assert summary["damage_dealt"] == 42
    assert summary["hp_start"] == 50
    assert summary["hp_end"] == 30
    assert summary["won_fight"] is True
    assert summary["state_type"] == "elite"


def test_killing_enemy_on_combat_death():
    pipe = DataPipeline()
    state = {
        "state_type": "elite",
        "player": {"hp": 0},
        "battle": {
            "enemies": [
                {
                    "name": "Jaw Worm",
                    "entity_id": "JW_0",
                    "hp": 10,
                    "intents": [{"type": "Attack", "damage": 11}],
                }
            ]
        },
    }
    with patch("sts2_agent.enemy_compendium.compact_enemy_intent") as compact:
        compact.return_value = {
            "intent": "Attack",
            "damage": 11,
            "compendium_key": "jaw_worm",
            "tags": ["attack"],
        }
        killer = pipe._infer_killing_enemy(state)

    assert killer is not None
    assert killer["name"] == "Jaw Worm"
    assert killer["entity_id"] == "JW_0"
    assert killer["compendium_key"] == "jaw_worm"


def test_extract_potion_belt_at_death_sparse_slots():
    from sts2_agent.data_pipeline import extract_potion_belt_at_death

    state = {
        "player": {
            "max_potion_slots": 3,
            "potions": [{"name": "Fire Potion"}, False, {"id": "BLOCK_POTION"}],
        }
    }
    max_slots, slots = extract_potion_belt_at_death(state)
    assert max_slots == 3
    assert slots[0] == "Fire Potion"
    assert slots[1] is None
    assert slots[2] == "BLOCK_POTION"


def test_record_decision_includes_card_reward_offered():
    pipe = DataPipeline()
    pipe._run_active = True
    pipe.run_id = "test-run"
    state = {
        "state_type": "card_reward",
        "run": {"floor": 5, "act": 1},
        "player": {},
        "card_reward": {
            "cards": [
                {"name": "Strike"},
                {"name": "Defend"},
                {"name": "Bash"},
            ]
        },
    }
    pipe.record_decision(state, {"action": "select_card_reward", "card_index": 0}, ["pick"])
    assert len(pipe._buffer) == 1
    row = pipe._buffer[0]
    assert row["card_reward_offered"] == ["Strike", "Defend", "Bash"]
    assert row["state_snapshot"]["card_reward_offered"] == ["Strike", "Defend", "Bash"]
    assert row["card_reward_picked"] == "Strike"
    assert row["state_snapshot"]["card_reward_picked"] == "Strike"
