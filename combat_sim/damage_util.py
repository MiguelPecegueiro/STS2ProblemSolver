"""Shared damage / vulnerable / slow helpers (engine + tuple DP)."""

from __future__ import annotations

VULNERABLE_MULT = 1.5
WEAK_MULT = 0.75
FRAIL_MULT = 0.75
SLOW_DAMAGE_PER_CARD = 0.10


def calc_damage(
    base: int,
    strength: int,
    vuln_stacks: int,
    slow_stacks: int = 0,
) -> int:
    """Attack damage: floor((base + str) * (1 + 0.1 * slow) * vuln_mult)."""
    modified = base + strength
    slow_mult = 1 + SLOW_DAMAGE_PER_CARD * slow_stacks
    vuln_mult = VULNERABLE_MULT if vuln_stacks > 0 else 1.0
    return int(modified * slow_mult * vuln_mult)


def apply_weak_to_damage(damage: int, weak_stacks: int) -> int:
    """Enemy outgoing attack damage (Weak on enemy)."""
    if damage <= 0 or weak_stacks <= 0:
        return max(0, damage)
    return int(damage * WEAK_MULT)


def apply_frail_to_block(block: int, frail_stacks: int) -> int:
    """Player block gained (Frail on player)."""
    if block <= 0 or frail_stacks <= 0:
        return block
    return int(block * FRAIL_MULT)


def damage_with_vulnerable(base: int, vuln_stacks: int) -> int:
    return calc_damage(base, 0, vuln_stacks, 0)


def damage_with_strength_and_vulnerable(
    base: int,
    strength: int,
    vuln_stacks: int,
) -> int:
    return calc_damage(base, strength, vuln_stacks, 0)


def apply_slow(damage: int, cards_played_before: int) -> int:
    """Legacy: apply slow multiplier to pre-scaled damage (prefer calc_damage)."""
    if cards_played_before <= 0:
        return damage
    return int(damage * (1 + SLOW_DAMAGE_PER_CARD * cards_played_before))


def optimistic_turn_attack_damage(
    *,
    strikes: int,
    defends: int,
    bash: int,
    bloodletting: int,
    inflame: int,
    strength: int,
    vuln_stacks: int,
    bash_damage: int,
    strike_damage: int,
) -> int:
    """Upper bound: BL -> Defend -> Inflame -> Bash -> Strike (maximizes slow on attacks)."""
    slow_n = 0
    total = 0
    for _ in range(bloodletting):
        slow_n += 1
    slow_n += defends + inflame
    for _ in range(bash):
        total += calc_damage(bash_damage, strength, vuln_stacks, slow_n)
        slow_n += 1
    for _ in range(strikes):
        total += calc_damage(strike_damage, strength, vuln_stacks, slow_n)
        slow_n += 1
    return total


def apply_damage_to_enemy_hp_block(
    enemy_hp: int,
    enemy_block: int,
    damage: int,
    *,
    hp_loss_cap: int | None = None,
    hp_lost_this_turn: int = 0,
) -> tuple[int, int, int]:
    """Apply damage; return (hp, block, hp_damage_dealt). Honors Hardened Shell cap."""
    if hp_loss_cap is not None:
        damage = min(damage, max(0, hp_loss_cap - hp_lost_this_turn))
    hp_before = enemy_hp
    remaining = damage
    if enemy_block > 0:
        absorbed = min(enemy_block, remaining)
        enemy_block -= absorbed
        remaining -= absorbed
    if remaining > 0:
        enemy_hp = max(0, enemy_hp - remaining)
    return enemy_hp, enemy_block, max(0, hp_before - enemy_hp)


def max_kill_damage_per_turn(raw_per_turn: int, hp_loss_cap_per_turn: int | None) -> int:
    """Per-turn cap for kill bound (Hardened Shell: at most cap HP removed per player turn)."""
    if hp_loss_cap_per_turn is None:
        return raw_per_turn
    return min(raw_per_turn, hp_loss_cap_per_turn)


def max_kill_total(
    raw_per_turn: int,
    turns_left: int,
    hp_loss_cap_per_turn: int | None = None,
) -> int:
    """Admissible max HP damage to enemy over remaining player turns."""
    if turns_left <= 0:
        return 0
    return max_kill_damage_per_turn(raw_per_turn, hp_loss_cap_per_turn) * turns_left


def min_turns_to_kill_shell(hp_e: int, hp_loss_cap_per_turn: int) -> int:
    """Minimum player turns to remove hp_e HP when each turn deals at most cap damage."""
    if hp_e <= 0:
        return 0
    if hp_loss_cap_per_turn <= 0:
        return 0
    return (hp_e + hp_loss_cap_per_turn - 1) // hp_loss_cap_per_turn


def prune_kill_shell_turn_budget(
    hp_e: int,
    turn: int,
    max_turns: int,
    hp_loss_cap_per_turn: int | None,
) -> bool:
    """True if enemy cannot die within remaining turns under Shell (admissible kill prune)."""
    if hp_e <= 0 or hp_loss_cap_per_turn is None:
        return False
    turns_left = max(0, max_turns - turn + 1)
    return min_turns_to_kill_shell(hp_e, hp_loss_cap_per_turn) > turns_left


def apply_damage_to_enemy_hp_block_legacy(
    enemy_hp: int,
    enemy_block: int,
    damage: int,
) -> tuple[int, int]:
    hp, block, _ = apply_damage_to_enemy_hp_block(enemy_hp, enemy_block, damage)
    return hp, block
