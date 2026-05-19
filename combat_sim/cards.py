"""Card definitions for the minimal combat simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CardDef:
    card_id: str
    name: str
    cost: int
    damage: int = 0
    block: int = 0
    vulnerable_apply: int = 0
    weak_apply: int = 0
    frail_apply: int = 0
    hp_loss: int = 0
    energy_gain: int = 0
    strength_apply: int = 0
    exhaust: bool = False
    hits: int = 1


def strike() -> CardDef:
    return CardDef("STRIKE", "Strike", cost=1, damage=6)


def defend() -> CardDef:
    return CardDef("DEFEND", "Defend", cost=1, block=5)


def bash() -> CardDef:
    return CardDef("BASH", "Bash", cost=2, damage=8, vulnerable_apply=2)


def twin_strike() -> CardDef:
    return CardDef("TWIN_STRIKE", "Twin Strike", cost=1, damage=5, hits=2)


def uppercut() -> CardDef:
    return CardDef(
        "UPPERCUT",
        "Uppercut",
        cost=2,
        damage=13,
        weak_apply=1,
        vulnerable_apply=1,
    )


# Sim 0: no Bash.
IRONCLAD_STARTER_STRIKE_DEFEND: tuple[CardDef, ...] = (strike(),) * 5 + (defend(),) * 4

def bloodletting() -> CardDef:
    return CardDef(
        "BLOODLETTING",
        "Bloodletting",
        cost=0,
        hp_loss=3,
        energy_gain=2,
    )


# Sim 1: full Ironclad starter.
IRONCLAD_STARTER_SIM1: tuple[CardDef, ...] = (strike(),) * 5 + (defend(),) * 4 + (bash(),)

def inflame() -> CardDef:
    return CardDef(
        "INFLAME",
        "Inflame",
        cost=1,
        strength_apply=2,
        exhaust=True,
    )


# Sim 2: starter + Bloodletting.
IRONCLAD_STARTER_SIM2: tuple[CardDef, ...] = IRONCLAD_STARTER_SIM1 + (bloodletting(),)

# Sim 3: + Inflame (permanent Strength, exhaust).
IRONCLAD_STARTER_SIM3: tuple[CardDef, ...] = IRONCLAD_STARTER_SIM2 + (inflame(),)

# General DP: Sim 3 deck + Twin Strike (13 cards).
IRONCLAD_STARTER_SIM3_TWIN_STRIKE: tuple[CardDef, ...] = IRONCLAD_STARTER_SIM3 + (twin_strike(),)
