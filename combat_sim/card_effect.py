"""Card definitions as data — state transition descriptors for the general engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CardTarget(str, Enum):
    ENEMY = "enemy"
    SELF = "self"
    ALL_ENEMIES = "all_enemies"


@dataclass(frozen=True, slots=True)
class CardEffect:
    """Deterministic immediate effects for one card play (Category 1)."""

    card_id: str = ""
    name: str = ""
    cost: int = 0
    damage: int = 0
    hits: int = 1
    block: int = 0
    hp_loss: int = 0
    energy_gain: int = 0
    vuln_apply: int = 0
    weak_apply: int = 0
    frail_apply: int = 0
    strength_apply: int = 0
    exhaust: bool = False
    draw: int = 0
    target: CardTarget = CardTarget.ENEMY
    # Phase 2+
    random_exhaust: int = 0
    approximate: bool = False
    triggers: tuple[str, ...] = field(default_factory=tuple)


# --- Starter deck as data (Ironclad) ---

STRIKE_EFFECT = CardEffect(card_id="STRIKE", name="Strike", cost=1, damage=6)
DEFEND_EFFECT = CardEffect(card_id="DEFEND", name="Defend", cost=1, block=5, target=CardTarget.SELF)
BASH_EFFECT = CardEffect(
    card_id="BASH", name="Bash", cost=2, damage=8, vuln_apply=2, target=CardTarget.ENEMY
)
BLOODLETTING_EFFECT = CardEffect(
    card_id="BLOODLETTING",
    name="Bloodletting",
    cost=0,
    hp_loss=3,
    energy_gain=2,
    target=CardTarget.SELF,
)
INFLAME_EFFECT = CardEffect(
    card_id="INFLAME",
    name="Inflame",
    cost=1,
    strength_apply=2,
    exhaust=True,
    target=CardTarget.SELF,
)
TWIN_STRIKE_EFFECT = CardEffect(
    card_id="TWIN_STRIKE", name="Twin Strike", cost=1, damage=5, hits=2
)
UPPERCUT_EFFECT = CardEffect(
    card_id="UPPERCUT",
    name="Uppercut",
    cost=2,
    damage=13,
    weak_apply=1,
    vuln_apply=1,
)

IRONCLAD_STARTER_SIM3_TWIN_STRIKE_EFFECTS: tuple[CardEffect, ...] = (
    *(
        STRIKE_EFFECT,
        STRIKE_EFFECT,
        STRIKE_EFFECT,
        STRIKE_EFFECT,
        STRIKE_EFFECT,
        DEFEND_EFFECT,
        DEFEND_EFFECT,
        DEFEND_EFFECT,
        DEFEND_EFFECT,
        BASH_EFFECT,
        BLOODLETTING_EFFECT,
        INFLAME_EFFECT,
    ),
    TWIN_STRIKE_EFFECT,
)

IRONCLAD_STARTER_EFFECTS: tuple[CardEffect, ...] = (
    STRIKE_EFFECT,
    STRIKE_EFFECT,
    STRIKE_EFFECT,
    STRIKE_EFFECT,
    STRIKE_EFFECT,
    DEFEND_EFFECT,
    DEFEND_EFFECT,
    DEFEND_EFFECT,
    DEFEND_EFFECT,
    BASH_EFFECT,
    BLOODLETTING_EFFECT,
    INFLAME_EFFECT,
)


def card_effect_from_card_def(definition) -> CardEffect:
    """Bridge legacy CardDef → CardEffect."""
    from combat_sim.cards import CardDef

    if not isinstance(definition, CardDef):
        raise TypeError(f"expected CardDef, got {type(definition).__name__}")
    target = CardTarget.ENEMY if definition.damage > 0 else CardTarget.SELF
    return CardEffect(
        card_id=definition.card_id,
        name=definition.name,
        cost=definition.cost,
        damage=definition.damage,
        hits=definition.hits,
        block=definition.block,
        hp_loss=definition.hp_loss,
        energy_gain=definition.energy_gain,
        vuln_apply=definition.vulnerable_apply,
        weak_apply=definition.weak_apply,
        frail_apply=definition.frail_apply,
        strength_apply=definition.strength_apply,
        exhaust=definition.exhaust,
        target=target,
    )
