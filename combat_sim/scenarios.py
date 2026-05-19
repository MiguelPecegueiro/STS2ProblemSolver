"""Preset fights for tests and experiments."""

from __future__ import annotations

from combat_sim.cards import (
    IRONCLAD_STARTER_SIM1,
    IRONCLAD_STARTER_SIM2,
    IRONCLAD_STARTER_SIM3,
    IRONCLAD_STARTER_SIM3_TWIN_STRIKE,
    IRONCLAD_STARTER_STRIKE_DEFEND,
)
from combat_sim.pattern_util import pattern_from_intents
from combat_sim.state import CombatState, EnemyState, Intent, IntentKind

# Bygone Effigy: Sleep 0 → Awaken 0 → Slashes 23 (effective). Innate Slow on enemy.
# HP 127 from Spire Codex (Act 1 elite, ascension 0).
BYGONE_EFFIGY_HP = 127
BYGONE_EFFIGY_PATTERN: list[Intent] = [
    Intent(IntentKind.ATTACK, 0),
    Intent(IntentKind.ATTACK, 0),
    Intent(IntentKind.ATTACK, 23),
]


def jaw_worm(*, hp: int = 40, seed: int = 42) -> CombatState:
    """Act-1 style single enemy: attack / block / heavy attack."""
    enemy = EnemyState(
        enemy_id="jaw_worm",
        name="Jaw Worm",
        hp=hp,
        max_hp=hp,
        pattern=[
            Intent(IntentKind.ATTACK, 7),
            Intent(IntentKind.BLOCK, 5),
            Intent(IntentKind.ATTACK, 11),
        ],
    )
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_STRIKE_DEFEND),
        player_hp=80,
        enemies=[enemy],
        seed=seed,
    )


def jaw_worm_sim1(*, hp: int = 40, seed: int = 42) -> CombatState:
    """Sim 1: full starter deck vs Jaw Worm."""
    enemy = EnemyState(
        enemy_id="jaw_worm",
        name="Jaw Worm",
        hp=hp,
        max_hp=hp,
        pattern=[
            Intent(IntentKind.ATTACK, 7),
            Intent(IntentKind.BLOCK, 5),
            Intent(IntentKind.ATTACK, 11),
        ],
    )
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_SIM1),
        player_hp=80,
        enemies=[enemy],
        seed=seed,
    )


def jaw_worm_sim2(*, hp: int = 40, seed: int = 42) -> CombatState:
    """Sim 2: starter + Bloodletting vs Jaw Worm."""
    enemy = EnemyState(
        enemy_id="jaw_worm",
        name="Jaw Worm",
        hp=hp,
        max_hp=hp,
        pattern=[
            Intent(IntentKind.ATTACK, 7),
            Intent(IntentKind.BLOCK, 5),
            Intent(IntentKind.ATTACK, 11),
        ],
    )
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_SIM2),
        player_hp=80,
        enemies=[enemy],
        seed=seed,
    )


def _bygone_effigy_enemy(*, hp: int = BYGONE_EFFIGY_HP) -> EnemyState:
    return EnemyState(
        enemy_id="bygone_effigy",
        name="Bygone Effigy",
        hp=hp,
        max_hp=hp,
        has_slow=True,
        pattern=list(BYGONE_EFFIGY_PATTERN),
    )


def bygone_effigy_sim0(*, hp: int = BYGONE_EFFIGY_HP, seed: int = 42) -> CombatState:
    """Strike/Defend vs Bygone Effigy (Tier A pattern)."""
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_STRIKE_DEFEND),
        player_hp=80,
        enemies=[_bygone_effigy_enemy(hp=hp)],
        seed=seed,
    )


def bygone_effigy_sim1(*, hp: int = BYGONE_EFFIGY_HP, seed: int = 42) -> CombatState:
    """Sim 1 deck (+ Bash) vs Bygone Effigy."""
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_SIM1),
        player_hp=80,
        enemies=[_bygone_effigy_enemy(hp=hp)],
        seed=seed,
    )


def bygone_effigy_sim2(*, hp: int = BYGONE_EFFIGY_HP, seed: int = 42) -> CombatState:
    """Sim 2 deck (+ Bloodletting) vs Act 1 elite Bygone Effigy (Tier A)."""
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_SIM2),
        player_hp=80,
        enemies=[_bygone_effigy_enemy(hp=hp)],
        seed=seed,
    )


def twin_strike_jaw_worm_sim3(*, hp: int = 40, seed: int = 42) -> CombatState:
    """Sim 3 deck + Twin Strike vs Jaw Worm (general DP multi-hit validation)."""
    enemy = EnemyState(
        enemy_id="jaw_worm",
        name="Jaw Worm",
        hp=hp,
        max_hp=hp,
        pattern=[
            Intent(IntentKind.ATTACK, 7),
            Intent(IntentKind.BLOCK, 5),
            Intent(IntentKind.ATTACK, 11),
        ],
    )
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_SIM3_TWIN_STRIKE),
        player_hp=80,
        enemies=[enemy],
        seed=seed,
    )


def jaw_worm_sim3(*, hp: int = 40, seed: int = 42) -> CombatState:
    """Sim 3: starter + Bloodletting + Inflame vs Jaw Worm."""
    enemy = EnemyState(
        enemy_id="jaw_worm",
        name="Jaw Worm",
        hp=hp,
        max_hp=hp,
        pattern=[
            Intent(IntentKind.ATTACK, 7),
            Intent(IntentKind.BLOCK, 5),
            Intent(IntentKind.ATTACK, 11),
        ],
    )
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_SIM3),
        player_hp=80,
        enemies=[enemy],
        seed=seed,
    )


def bygone_effigy_sim3(*, hp: int = BYGONE_EFFIGY_HP, seed: int = 42) -> CombatState:
    """Sim 3 deck (+ Inflame) vs Bygone Effigy (Tier A)."""
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_SIM3),
        player_hp=80,
        enemies=[_bygone_effigy_enemy(hp=hp)],
        seed=seed,
    )


# Skulking Colony (Underdocks Act 1 elite). Cycle: Inertia 12 → Zoom 14+10blk → Inertia 9 → Stabs 14.
SKULKING_COLONY_HP = 70
SKULKING_COLONY_PATTERN: list[Intent] = [
    Intent(IntentKind.ATTACK, 12),
    Intent(IntentKind.ATTACK, 14, enemy_block_bonus=10),
    Intent(IntentKind.ATTACK, 9),
    Intent(IntentKind.ATTACK, 14),
]
SKULKING_COLONY_PATTERN_TUPLE = pattern_from_intents(SKULKING_COLONY_PATTERN)


def _skulking_colony_enemy(*, hp: int = SKULKING_COLONY_HP) -> EnemyState:
    return EnemyState(
        enemy_id="skulking_colony",
        name="Skulking Colony",
        hp=hp,
        max_hp=hp,
        hp_loss_cap_per_turn=15,
        pattern=list(SKULKING_COLONY_PATTERN),
    )


def skulking_colony_sim3(*, hp: int = SKULKING_COLONY_HP, seed: int = 42) -> CombatState:
    """Sim 3 deck vs Skulking Colony (Hardened Shell + attack cycle)."""
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_SIM3),
        player_hp=80,
        enemies=[_skulking_colony_enemy(hp=hp)],
        seed=seed,
    )


def slime_boss_minion(*, hp: int = 12, seed: int = 7) -> CombatState:
    """Weak attacker for lethal and block tests."""
    enemy = EnemyState(
        enemy_id="slime",
        name="Acid Slime (S)",
        hp=hp,
        max_hp=hp,
        pattern=[Intent(IntentKind.ATTACK, 8)],
    )
    return CombatState.new_fight(
        deck=list(IRONCLAD_STARTER_STRIKE_DEFEND),
        player_hp=50,
        enemies=[enemy],
        seed=seed,
    )
