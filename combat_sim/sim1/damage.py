"""Vulnerable damage math for Sim 1 (re-exports shared helpers)."""

from __future__ import annotations

from combat_sim.damage_util import (
    apply_damage_to_enemy_hp_block_legacy as apply_damage_to_enemy,
    damage_with_vulnerable,
)

STRIKE_DAMAGE = 6
BASH_DAMAGE = 8
VULNERABLE_MULT = 1.5

__all__ = [
    "STRIKE_DAMAGE",
    "BASH_DAMAGE",
    "VULNERABLE_MULT",
    "apply_damage_to_enemy",
    "damage_with_vulnerable",
]
