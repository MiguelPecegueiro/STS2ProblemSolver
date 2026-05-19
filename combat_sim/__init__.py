"""Minimal Slay the Spire combat simulator (damage + block only)."""

from combat_sim.cards import CardDef, IRONCLAD_STARTER_STRIKE_DEFEND, strike, defend
from combat_sim.engine import CombatEngine, FightResult, TurnAction
from combat_sim.scenarios import jaw_worm, slime_boss_minion
from combat_sim.solver import solve_fight, solve_turn
from combat_sim.state import CombatPhase, CombatState, EnemyState, Intent, IntentKind

__all__ = [
    "CardDef",
    "CombatEngine",
    "CombatPhase",
    "CombatState",
    "EnemyState",
    "FightResult",
    "Intent",
    "IntentKind",
    "IRONCLAD_STARTER_STRIKE_DEFEND",
    "TurnAction",
    "defend",
    "jaw_worm",
    "slime_boss_minion",
    "solve_fight",
    "solve_turn",
    "strike",
]
