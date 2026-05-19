"""Convert intents / pattern steps for tuple DP and engine."""

from __future__ import annotations

from combat_sim.state import Intent, IntentKind

# ("A"|"B", value) or ("A", value, enemy_block_bonus_on_this_step)
PatternStep = tuple[str, int] | tuple[str, int, int]
Pattern = tuple[PatternStep, ...]


def parse_pattern_step(step: PatternStep) -> tuple[str, int, int]:
    kind = step[0]
    val = step[1]
    bonus_block = step[2] if len(step) > 2 else 0
    return kind, val, bonus_block


def pattern_from_intents(intents: list[Intent]) -> Pattern:
    steps: list[PatternStep] = []
    for intent in intents:
        if intent.kind == IntentKind.ATTACK:
            if intent.enemy_block_bonus > 0:
                steps.append(("A", intent.value, intent.enemy_block_bonus))
            else:
                steps.append(("A", intent.value))
        else:
            steps.append(("B", intent.value))
    return tuple(steps)
