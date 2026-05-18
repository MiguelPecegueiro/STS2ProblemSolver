"""Combat potion evaluation and emergency thresholds."""

from __future__ import annotations

from sts2_agent.potions import (
    CRITICAL_HEAL_HP_RATIO,
    EMERGENCY_HP_RATIO,
    CombatPotionContext,
    clear_potion_session_failures,
    clear_potion_use_failures,
    emergency_potion_score,
    evaluate_combat_potions,
    get_potion_profile,
    is_potion_slot_failed,
    mark_potion_use_failed,
    note_potion_use_no_effect,
    player_hp_ratio,
    potion_needs_enemy_target,
)


def _ctx(**overrides) -> CombatPotionContext:
    base = dict(
        player={
            "hp": 20,
            "max_hp": 80,
            "potions": [
                {"slot": 0, "name": "Fruit Juice", "id": "FRUIT_JUICE", "can_use_in_combat": True},
            ],
        },
        enemies=[{"hp": 30, "entity_id": "e1"}],
        incoming=10,
        block=0,
        lethal_target=None,
        lethal_damage=0,
        has_playable_cards=True,
        hp=20,
        max_hp=80,
        hp_ratio=0.25,
        gap=10,
    )
    base.update(overrides)
    return CombatPotionContext(**base)


def test_player_hp_ratio_alt_fields() -> None:
    hp, max_hp, ratio = player_hp_ratio({"current_hp": 18, "maxHp": 72})
    assert hp == 18
    assert max_hp == 72
    assert abs(ratio - 0.25) < 0.001


def test_low_hp_uses_heal_at_25_percent() -> None:
    clear_potion_use_failures(_ctx().player)
    action, reasons = evaluate_combat_potions(_ctx())
    assert action == {"action": "use_potion", "slot": 0}
    assert any("critical HP" in r for r in reasons)


def test_emergency_uses_block_when_no_heal() -> None:
    clear_potion_use_failures(_ctx().player)
    ctx = _ctx(
        player={
            "hp": 20,
            "max_hp": 80,
            "potions": [{"slot": 0, "name": "Block Potion", "id": "BLOCK_POTION"}],
        },
    )
    action, reasons = evaluate_combat_potions(ctx)
    assert action == {"action": "use_potion", "slot": 0}
    assert any("emergency" in r for r in reasons)


def test_emergency_does_not_fire_above_threshold() -> None:
    clear_potion_use_failures(_ctx().player)
    ctx = _ctx(hp=30, max_hp=80, hp_ratio=30 / 80, gap=3, incoming=3, block=0)
    action, reasons = evaluate_combat_potions(ctx)
    assert action is None
    assert any("no use" in r for r in reasons)
    assert any(f"{EMERGENCY_HP_RATIO:.0%}" in r for r in reasons)


def test_critical_heal_before_emergency_band() -> None:
    clear_potion_use_failures(_ctx().player)
    ctx = _ctx(hp=27, max_hp=80, hp_ratio=27 / 80)
    action, reasons = evaluate_combat_potions(ctx)
    assert action == {"action": "use_potion", "slot": 0}
    assert any("critical HP" in r for r in reasons)


def test_emergency_logs_scores_for_each_slot() -> None:
    clear_potion_use_failures(_ctx().player)
    ctx = _ctx(
        player={
            "hp": 10,
            "max_hp": 80,
            "potions": [
                {"slot": 0, "name": "Fire Potion", "id": "FIRE_POTION"},
                {"slot": 1, "name": "Block Potion", "id": "BLOCK_POTION"},
            ],
        },
        hp=10,
        hp_ratio=10 / 80,
    )
    action, reasons = evaluate_combat_potions(ctx)
    assert action is not None
    assert any("emergency score=" in r for r in reasons)


def test_emergency_threshold_constants() -> None:
    assert CRITICAL_HEAL_HP_RATIO == 0.35
    assert EMERGENCY_HP_RATIO == 0.25


def test_api_reject_blocks_empty_slot_after_belt_clears() -> None:
    """Failed-slot memory must survive belt fingerprint change (empty belt)."""
    clear_potion_session_failures()
    player_with = {
        "max_potion_slots": 3,
        "potions": [{"slot": 0, "name": "Fire Potion", "id": "FIRE_POTION"}],
    }
    mark_potion_use_failed(player_with, 0)
    player_empty = {"max_potion_slots": 3, "potions": []}
    assert is_potion_slot_failed(player_empty, 0)
    clear_potion_session_failures()


def test_note_potion_use_no_effect_blocks_slot() -> None:
    player = {
        "hp": 10,
        "max_hp": 80,
        "potions": [{"slot": 0, "name": "Flex Potion", "id": "FLEX_POTION"}],
    }
    before = {"player": player}
    after = {"player": dict(player)}
    action = {"action": "use_potion", "slot": 0, "target": "FOGMOG_0"}
    assert note_potion_use_no_effect(before, after, action) is True
    assert is_potion_slot_failed(player, 0)
    ctx = _ctx(player=player, hp=10, hp_ratio=10 / 80)
    assert evaluate_combat_potions(ctx)[0] is None


def test_emergency_buff_potion_omits_enemy_target() -> None:
    """Flex-style buffs must not inherit lethal_target (causes API no-op loops)."""
    player = {
        "hp": 10,
        "max_hp": 80,
        "potions": [{"slot": 0, "name": "Flex Potion", "id": "FLEX_POTION"}],
    }
    clear_potion_use_failures(player)
    ctx = _ctx(
        player=player,
        hp=10,
        hp_ratio=10 / 80,
        lethal_target={"hp": 50, "entity_id": "FOGMOG_0", "name": "Fogmog"},
    )
    action, reasons = evaluate_combat_potions(ctx)
    assert action == {"action": "use_potion", "slot": 0}
    assert "target" not in action
    assert any("emergency" in r for r in reasons)


def test_offensive_potion_includes_enemy_target() -> None:
    clear_potion_use_failures(_ctx().player)
    ctx = _ctx(
        player={
            "hp": 10,
            "max_hp": 80,
            "potions": [
                {
                    "slot": 0,
                    "name": "Fire Potion",
                    "id": "FIRE_POTION",
                    "description": "Deal 20 damage to an enemy.",
                }
            ],
        },
        hp=10,
        hp_ratio=10 / 80,
        enemies=[{"hp": 50, "entity_id": "BOSS_0", "name": "Boss"}],
        lethal_target={"hp": 50, "entity_id": "BOSS_0"},
    )
    profile = get_potion_profile(ctx.player["potions"][0])
    assert potion_needs_enemy_target(ctx.player["potions"][0], profile)
    action, reasons = evaluate_combat_potions(ctx)
    assert action == {"action": "use_potion", "slot": 0, "target": "BOSS_0"}
    assert any("lethal setup" in r or "emergency" in r for r in reasons)


def test_debuff_potion_includes_target() -> None:
    clear_potion_use_failures(_ctx().player)
    ctx = _ctx(
        hp=35,
        max_hp=80,
        hp_ratio=35 / 80,
        gap=8,
        incoming=12,
        block=4,
        player={
            "hp": 35,
            "max_hp": 80,
            "potions": [
                {
                    "slot": 0,
                    "name": "Weak Potion",
                    "id": "WEAK_POTION",
                    "description": "Apply 3 Weak to an enemy.",
                }
            ],
        },
        enemies=[{"hp": 40, "entity_id": "SLIME_0"}],
    )
    action, _ = evaluate_combat_potions(ctx)
    assert action == {"action": "use_potion", "slot": 0, "target": "SLIME_0"}


def test_failed_potion_slot_skipped() -> None:
    player = {
        "hp": 10,
        "max_hp": 80,
        "potions": [{"slot": 0, "name": "Fire Potion", "id": "FIRE_POTION"}],
    }
    mark_potion_use_failed(player, 0)
    assert is_potion_slot_failed(player, 0)
    ctx = _ctx(player=player, hp=10, hp_ratio=10 / 80)
    action, reasons = evaluate_combat_potions(ctx)
    assert action is None
    assert any("blocked after API reject" in r for r in reasons)


def test_unclassified_potion_gets_emergency_fallback() -> None:
    profile = get_potion_profile({"name": "Mystery Brew", "id": "UNKNOWN_XYZ"})
    assert emergency_potion_score(profile, gap=10, has_playable_cards=False) == 0
    clear_potion_use_failures(_ctx().player)
    ctx = _ctx(
        player={"hp": 10, "max_hp": 80, "potions": [{"slot": 0, "name": "Mystery Brew"}]},
        hp=10,
        hp_ratio=10 / 80,
    )
    action, reasons = evaluate_combat_potions(ctx)
    assert action == {"action": "use_potion", "slot": 0}
    assert any("fallback" in r for r in reasons)
