"""Combat potion evaluation and emergency thresholds."""

from __future__ import annotations

from sts2_agent.potions import (
    CRITICAL_HEAL_HP_RATIO,
    EMERGENCY_HP_RATIO,
    CombatPotionContext,
    clear_potion_use_failures,
    emergency_potion_score,
    evaluate_combat_potions,
    get_potion_profile,
    player_hp_ratio,
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
