"""Potion belt helpers, scoring, and STS2 drop-chance heuristics."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from sts2_agent.knowledge import KnowledgeBase, get_knowledge

logger = logging.getLogger(__name__)

# Combat drink thresholds (see evaluate_combat_potions).
CRITICAL_HEAL_HP_RATIO = 0.35
EMERGENCY_HP_RATIO = 0.25

# Post-combat potion chance at act start; ±10% per drop/no-drop; elites +12.5%.
ACT_BASE_DROP_CHANCE = 0.40
DROP_CHANCE_STEP = 0.10
ELITE_DROP_BONUS = 0.125


class PotionDropTracker:
    """Track estimated post-combat potion offer chance within an act."""

    def __init__(self) -> None:
        self._act: int = 0
        self._estimated_chance: float = ACT_BASE_DROP_CHANCE
        self._last_combat_saw_potion_offer: bool = False
        self._next_rewards_elite_bonus: bool = False

    def reset_for_act(self, act: int) -> None:
        if act != self._act:
            self._act = act
            self._estimated_chance = ACT_BASE_DROP_CHANCE

    def note_combat_ended(self, state_type: str) -> None:
        self._next_rewards_elite_bonus = state_type.lower() == "elite"

    def estimated_drop_chance(self, state: dict, *, is_elite: bool = False) -> float:
        run = state.get("run") or {}
        act = int(run.get("act") or 1)
        self.reset_for_act(act)
        chance = self._estimated_chance
        elite = is_elite or self._next_rewards_elite_bonus
        if elite:
            chance += ELITE_DROP_BONUS
        return max(0.05, min(0.95, chance))

    def note_rewards_screen(self, state: dict, items: list[dict]) -> None:
        self._last_combat_saw_potion_offer = any(
            str(i.get("type") or "").lower() == "potion" for i in items
        )
        self._next_rewards_elite_bonus = False

    def note_potion_taken(self) -> None:
        self._estimated_chance = max(0.05, self._estimated_chance - DROP_CHANCE_STEP)

    def note_potion_skipped(self) -> None:
        self._estimated_chance = min(0.95, self._estimated_chance + DROP_CHANCE_STEP)

    def should_deprioritize_potion_offer(self, state: dict, *, is_elite: bool) -> bool:
        """When belt has space but drop odds are low, favor gold/relic over potion."""
        if potion_belt_full(state):
            return False
        chance = self.estimated_drop_chance(state, is_elite=is_elite)
        return chance < 0.35


_tracker = PotionDropTracker()


def get_potion_drop_tracker() -> PotionDropTracker:
    return _tracker


def potion_is_filled(potion: object) -> bool:
    """True when this belt entry is a real potion (not an empty slot placeholder)."""
    if potion is None or potion is False:
        return False
    if isinstance(potion, dict):
        if potion.get("empty") or potion.get("is_empty"):
            return False
        pid = str(potion.get("id") or potion.get("name") or "").strip()
        return bool(pid)
    text = str(potion).strip()
    if not text:
        return False
    return text.lower() not in ("null", "none", "empty", "false")


def potion_belt_full(state: dict) -> bool:
    player = state.get("player") or {}
    max_slots = int(player.get("max_potion_slots") or 3)
    filled = sum(1 for _slot, _ in iter_potion_belt_slots(player))
    return filled >= max_slots


def iter_potion_belt_slots(player: dict) -> list[tuple[int, object]]:
    """(belt_slot_index, potion) for each occupied slot - index matches use_potion slot."""
    max_slots = int(player.get("max_potion_slots") or 3)
    raw = player.get("potions")
    if not raw:
        return []

    if isinstance(raw, dict):
        out: list[tuple[int, object]] = []
        for key, potion in raw.items():
            if not potion_is_filled(potion):
                continue
            try:
                slot = int(key)
            except (TypeError, ValueError):
                continue
            out.append((slot, potion))
        return sorted(out, key=lambda x: x[0])

    if not isinstance(raw, list):
        return []

    has_gaps = len(raw) >= max_slots or any(
        entry is None or entry is False for entry in raw
    )

    out: list[tuple[int, object]] = []
    if has_gaps:
        for slot in range(max_slots):
            potion = raw[slot] if slot < len(raw) else None
            if potion_is_filled(potion):
                out.append((slot, potion))
        return out

    for list_idx, potion in enumerate(raw):
        if not potion_is_filled(potion):
            continue
        if isinstance(potion, dict):
            explicit = potion.get("slot")
            if explicit is None:
                explicit = potion.get("index")
            if explicit is not None:
                try:
                    out.append((int(explicit), potion))
                    continue
                except (TypeError, ValueError):
                    pass
        out.append((list_idx, potion))
    return out


def filled_potion_slots(player: dict) -> list[tuple[int, dict]]:
    slots: list[tuple[int, dict]] = []
    for slot, potion in iter_potion_belt_slots(player):
        if isinstance(potion, dict):
            slots.append((slot, potion))
        else:
            slots.append((slot, {"id": str(potion), "name": str(potion)}))
    return slots


def potion_combat_label(potion: object) -> str:
    if isinstance(potion, dict):
        name = str(potion.get("name") or "").lower()
        pid = str(potion.get("id") or "").lower()
        desc = str(potion.get("description") or "").lower()
        return f"{name} {pid} {desc}".strip()
    return str(potion or "").lower()


_failed_use_slots: dict[str, set[int]] = {}


def potion_use_fail_key(player: dict) -> str:
    parts = [str(player.get("max_potion_slots") or 3)]
    for slot, potion in iter_potion_belt_slots(player):
        if isinstance(potion, dict):
            parts.append(f"{slot}:{potion.get('id') or potion.get('name')}")
        else:
            parts.append(f"{slot}:{potion}")
    return "|".join(parts)


def mark_potion_use_failed(player: dict, slot: int) -> None:
    _failed_use_slots.setdefault(potion_use_fail_key(player), set()).add(int(slot))


def clear_potion_use_failures(player: dict) -> None:
    _failed_use_slots.pop(potion_use_fail_key(player), None)


def is_potion_slot_failed(player: dict, slot: int) -> bool:
    return int(slot) in _failed_use_slots.get(potion_use_fail_key(player), set())


def potion_label(potion: dict) -> str:
    return str(potion.get("name") or potion.get("id") or "").strip()


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def player_hp_ratio(player: dict) -> tuple[int, int, float]:
    """(current_hp, max_hp, ratio) - tolerates alternate API field names."""
    max_hp = _safe_int(player.get("max_hp"))
    if max_hp is None:
        max_hp = _safe_int(player.get("maxHp"))
    max_hp = max_hp or 1

    hp = _safe_int(player.get("hp"))
    if hp is None:
        hp = _safe_int(player.get("current_hp"))
    if hp is None:
        hp = _safe_int(player.get("currentHp"))
    if hp is None:
        hp = max_hp

    ratio = hp / max_hp if max_hp else 1.0
    return hp, max_hp, ratio


def potion_lookup_key(potion: object) -> str | None:
    if isinstance(potion, dict):
        for field in ("id", "name"):
            val = str(potion.get(field) or "").strip()
            if val:
                return val
        return None
    text = str(potion or "").strip()
    return text or None


def _strip_markup(text: str) -> str:
    """Normalize Codex / API potion description text for keyword rules."""
    cleaned = re.sub(r"\[[^\]]*\]", " ", text)
    cleaned = re.sub(r"\{[^}]*\}", " ", cleaned)
    return " ".join(cleaned.lower().split())


@dataclass(frozen=True)
class PotionProfile:
    heal: bool = False
    block: bool = False
    defensive: bool = False
    offensive: bool = False
    debuff: bool = False
    buff: bool = False
    draw: bool = False
    self_damage: bool = False
    passive: bool = False


def _classify_potion_text(desc: str) -> PotionProfile:
    text = _strip_markup(desc)
    if not text:
        return PotionProfile()

    passive = "when your hp would be reduced" in text
    self_damage = "all players" in text or "players and enemies" in text
    heal = "heal" in text or ("gain" in text and "max hp" in text) or "regen" in text
    block = any(
        k in text
        for k in ("block", "intangible", "buffer", "plating", "triple your block")
    )
    defensive = "less damage" in text
    offensive = "deal" in text and "damage" in text and not passive
    debuff = any(
        k in text for k in ("weak", "vulnerable", "poison", "doom", "demise")
    ) and ("enem" in text or "all enemies" in text or "enemy" in text)
    buff = "gain" in text and any(
        k in text for k in ("strength", "dexterity", "focus", "energy", "thorns")
    )
    draw = "draw" in text and "card" in text

    return PotionProfile(
        heal=heal,
        block=block,
        defensive=defensive,
        offensive=offensive,
        debuff=debuff,
        buff=buff,
        draw=draw,
        self_damage=self_damage,
        passive=passive,
    )


def _classify_potion_label_fallback(label: str) -> PotionProfile:
    text = label.lower()
    return PotionProfile(
        heal=any(k in text for k in ("heal", "fruit", "blood", "fairy", "regen", "essence")),
        block=any(
            k in text
            for k in ("block", "armor", "fossil", "stone", "shield", "smooth", "liquid bronze")
        ),
        offensive=any(
            k in text for k in ("fire", "explosive", "attack", "bomb", "poison", "fear", "foul")
        ),
        debuff=any(k in text for k in ("weak", "vulnerable", "binding", "poison")),
        buff=any(
            k in text
            for k in ("strength", "dexterity", "skill", "speed", "swift", "flex", "energy", "power")
        ),
        draw="swift" in text or "draw" in text,
        self_damage="foul" in text,
    )


def potion_needs_enemy_target(
    potion: object,
    profile: PotionProfile,
    kb: KnowledgeBase | None = None,
) -> bool:
    """True when API expects use_potion with a single enemy entity_id."""
    if isinstance(potion, dict):
        for key in ("requires_target", "needs_target", "requires_target_enemy"):
            if potion.get(key) is True:
                return True
        mode = str(potion.get("target_type") or potion.get("target") or "").lower().replace(
            " ", ""
        )
        if mode in ("enemy", "anyenemy", "singleenemy", "single_enemy"):
            return True
        if mode in ("none", "self", "all", "allenemies", "all_enemies", "everyenemy"):
            return False

    kb = kb or get_knowledge()
    key = potion_lookup_key(potion)
    codex = kb.lookup_potion(key) if key else None
    if codex:
        desc = _strip_markup(
            str(codex.get("description") or codex.get("description_raw") or "")
        )
        if "all enemies" in desc or "all enemy" in desc or "every enemy" in desc:
            return False
        if any(
            phrase in desc
            for phrase in (
                "target",
                "an enemy",
                "single enemy",
                "random enemy",
                "choose an enemy",
            )
        ):
            if profile.heal or profile.block or profile.defensive:
                return False
            return True

    if profile.offensive or profile.debuff:
        return True
    if profile.self_damage:
        return True
    return False


def pick_potion_target(
    ctx: CombatPotionContext,
    *,
    prefer: dict | None = None,
) -> dict | None:
    if prefer is not None and int(prefer.get("hp") or 0) > 0:
        return prefer
    living = [e for e in ctx.enemies if int(e.get("hp") or 0) > 0]
    if not living:
        return None
    from sts2_agent.scorer import pick_highest_threat_enemy

    target, _ = pick_highest_threat_enemy(living)
    return target


def get_potion_profile(
    potion: object,
    kb: KnowledgeBase | None = None,
) -> PotionProfile:
    """Combat/reward classification from Spire Codex description, with label fallback."""
    kb = kb or get_knowledge()
    key = potion_lookup_key(potion)
    codex = kb.lookup_potion(key) if key else None
    if codex:
        desc = str(codex.get("description") or codex.get("description_raw") or "")
        profile = _classify_potion_text(desc)
        if profile != PotionProfile():
            return profile
    if isinstance(potion, dict):
        desc = str(potion.get("description") or "")
        if desc:
            profile = _classify_potion_text(desc)
            if profile != PotionProfile():
                return profile
    return _classify_potion_label_fallback(potion_combat_label(potion))


@dataclass(frozen=True)
class CombatPotionContext:
    player: dict
    enemies: list[dict]
    incoming: int
    block: int
    lethal_target: dict | None
    lethal_damage: int
    has_playable_cards: bool
    hp: int
    max_hp: int
    hp_ratio: float
    gap: int


def _potion_skip_reason(
    slot: int,
    label: str,
    profile: PotionProfile,
    *,
    hp_ratio: float,
    gap: int,
    incoming: int,
    has_playable_cards: bool,
    lethal_target: dict | None,
    usable: bool,
) -> str:
    if not usable:
        return f"  slot {slot} {label}: skip - cannot use in combat"
    if profile.passive:
        return f"  slot {slot} {label}: skip - passive"
    if profile.heal and hp_ratio > CRITICAL_HEAL_HP_RATIO:
        return (
            f"  slot {slot} {label}: skip heal - HP {hp_ratio:.1%} > "
            f"{CRITICAL_HEAL_HP_RATIO:.0%} critical"
        )
    if (profile.block or profile.defensive) and (
        gap <= 8 or lethal_target or (has_playable_cards and gap <= 15)
    ):
        return (
            f"  slot {slot} {label}: skip defensive - gap={gap} "
            f"playable={has_playable_cards} lethal={bool(lethal_target)}"
        )
    if profile.offensive and not lethal_target:
        return f"  slot {slot} {label}: skip offensive - no lethal / weak enemy setup"
    if profile.debuff and (gap <= 5 or hp_ratio >= 0.5):
        return (
            f"  slot {slot} {label}: skip debuff - gap={gap} HP={hp_ratio:.1%}"
        )
    if (profile.buff or profile.draw) and (incoming > 0 or not has_playable_cards):
        return f"  slot {slot} {label}: skip tempo - incoming={incoming} playable={has_playable_cards}"
    return (
        f"  slot {slot} {label}: no proactive rule "
        f"[heal={profile.heal} block={profile.block} off={profile.offensive} "
        f"debuff={profile.debuff} buff={profile.buff} draw={profile.draw}]"
    )


def evaluate_combat_potions(ctx: CombatPotionContext) -> tuple[dict | None, list[str]]:
    """Score belt potions for combat; always returns diagnostic reasons."""
    kb = get_knowledge()
    reasons: list[str] = [
        f"potions: HP {ctx.hp}/{ctx.max_hp} ({ctx.hp_ratio:.1%}) "
        f"incoming={ctx.incoming} block={ctx.block} gap={ctx.gap} "
        f"playable_cards={ctx.has_playable_cards}"
    ]

    player = ctx.player
    belt = [
        (slot, potion)
        for slot, potion in iter_potion_belt_slots(player)
        if not is_potion_slot_failed(player, slot)
    ]
    failed = [
        slot
        for slot, _ in iter_potion_belt_slots(player)
        if is_potion_slot_failed(player, slot)
    ]
    if failed:
        reasons.append(f"potions: slots blocked after API reject: {failed}")

    if not belt:
        reasons.append("potions: belt empty - nothing to use")
        logger.debug("potions: empty belt (HP %.0f%%)", ctx.hp_ratio * 100)
        return None, reasons

    reasons.append(f"potions: {len(belt)} filled slot(s)")

    def _usable(slot: int, potion: object) -> bool:
        return not (
            isinstance(potion, dict) and potion.get("can_use_in_combat") is False
        )

    def _use_action(
        slot: int,
        potion: object,
        *,
        profile: PotionProfile,
        target: dict | None = None,
        rule: str,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        label = potion_label(potion) if isinstance(potion, dict) else potion_combat_label(potion)
        need_target = potion_needs_enemy_target(potion, profile, kb)
        resolved = target
        if need_target:
            resolved = pick_potion_target(ctx, prefer=target)
            if not resolved or not resolved.get("entity_id"):
                reasons.append(
                    f"  slot {slot} {label}: skip - needs enemy target (none available)"
                )
                return None, reasons

        action: dict[str, Any] = {"action": "use_potion", "slot": slot}
        entity = (resolved or {}).get("entity_id")
        if entity:
            action["target"] = entity
        msg = f"potions: USE slot {slot} ({label}) - {rule}"
        if entity:
            msg += f" -> {entity}"
        reasons.append(msg)
        logger.info(msg)
        return action, reasons

    living = [e for e in ctx.enemies if int(e.get("hp") or 0) > 0]

    for slot, potion in belt:
        label = potion_label(potion) if isinstance(potion, dict) else potion_combat_label(potion)
        if not _usable(slot, potion):
            reasons.append(f"  slot {slot} {label}: skip - cannot use in combat")
            continue
        profile = get_potion_profile(potion, kb)
        if profile.passive:
            reasons.append(f"  slot {slot} {label}: skip - passive")
            continue

        if ctx.hp_ratio <= CRITICAL_HEAL_HP_RATIO and profile.heal:
            action, use_reasons = _use_action(
                slot,
                potion,
                profile=profile,
                rule=f"critical HP ({ctx.hp_ratio:.1%}) heal",
            )
            if action:
                return action, use_reasons

        if (
            ctx.gap > 8
            and not ctx.lethal_target
            and (profile.block or profile.defensive)
            and (not ctx.has_playable_cards or ctx.gap > 15)
        ):
            action, use_reasons = _use_action(
                slot,
                potion,
                profile=profile,
                rule=f"incoming {ctx.incoming} vs block {ctx.block} defensive",
            )
            if action:
                return action, use_reasons

        if ctx.lethal_target and profile.offensive and not profile.self_damage:
            action, use_reasons = _use_action(
                slot,
                potion,
                profile=profile,
                target=ctx.lethal_target,
                rule="lethal setup offensive",
            )
            if action:
                return action, use_reasons

        if profile.offensive and not profile.self_damage:
            weak = [e for e in living if int(e.get("hp") or 0) <= 25]
            if weak and not ctx.has_playable_cards:
                target = min(weak, key=lambda e: int(e.get("hp") or 0))
                action, use_reasons = _use_action(
                    slot,
                    potion,
                    profile=profile,
                    target=target,
                    rule="finish low-HP enemy",
                )
                if action:
                    return action, use_reasons

        if profile.debuff and ctx.gap > 5 and ctx.hp_ratio < 0.5:
            action, use_reasons = _use_action(
                slot,
                potion,
                profile=profile,
                rule=f"debuff vs incoming gap {ctx.gap}",
            )
            if action:
                return action, use_reasons

        if (profile.buff or profile.draw) and ctx.incoming == 0 and ctx.has_playable_cards:
            action, use_reasons = _use_action(
                slot,
                potion,
                profile=profile,
                rule="tempo at combat start",
            )
            if action:
                return action, use_reasons

    if ctx.hp_ratio <= EMERGENCY_HP_RATIO:
        reasons.append(
            f"potions: emergency HP ({ctx.hp_ratio:.1%} <= {EMERGENCY_HP_RATIO:.0%})"
        )
        best: tuple[int, object, int] | None = None
        for slot, potion in belt:
            if not _usable(slot, potion):
                continue
            profile = get_potion_profile(potion, kb)
            label = potion_label(potion) if isinstance(potion, dict) else str(potion)
            score = emergency_potion_score(
                profile, gap=ctx.gap, has_playable_cards=ctx.has_playable_cards
            )
            reasons.append(
                f"  slot {slot} {label}: emergency score={score} "
                f"(heal={profile.heal} block={profile.block} passive={profile.passive})"
            )
            if score <= 0:
                continue
            if best is None or score > best[2]:
                best = (slot, potion, score)

        if best:
            slot, potion, score = best
            profile = get_potion_profile(potion, kb)
            action, use_reasons = _use_action(
                slot,
                potion,
                profile=profile,
                target=ctx.lethal_target,
                rule=f"emergency best score={score}",
            )
            if action:
                return action, use_reasons

        for slot, potion in belt:
            if not _usable(slot, potion):
                continue
            profile = get_potion_profile(potion, kb)
            label = potion_label(potion) if isinstance(potion, dict) else str(potion)
            if profile.passive or profile.self_damage:
                reasons.append(f"  slot {slot} {label}: emergency fallback skip")
                continue
            action, use_reasons = _use_action(
                slot,
                potion,
                profile=profile,
                target=ctx.lethal_target,
                rule="emergency fallback (any survivable)",
            )
            if action:
                return action, use_reasons

        reasons.append("potions: emergency - no usable survivable potion")
        logger.warning(
            "Emergency HP %.1f%% but no potion to drink (belt=%d)",
            ctx.hp_ratio * 100,
            len(belt),
        )
    else:
        for slot, potion in belt:
            label = potion_label(potion) if isinstance(potion, dict) else potion_combat_label(potion)
            usable = _usable(slot, potion)
            profile = get_potion_profile(potion, kb) if usable else PotionProfile()
            reasons.append(
                _potion_skip_reason(
                    slot,
                    label,
                    profile,
                    hp_ratio=ctx.hp_ratio,
                    gap=ctx.gap,
                    incoming=ctx.incoming,
                    has_playable_cards=ctx.has_playable_cards,
                    lethal_target=ctx.lethal_target,
                    usable=usable,
                )
            )
        reasons.append(
            f"potions: no use - HP {ctx.hp_ratio:.1%} above emergency "
            f"({EMERGENCY_HP_RATIO:.0%}) and no proactive rule matched"
        )

    return None, reasons


def emergency_potion_score(
    profile: PotionProfile,
    *,
    gap: int,
    has_playable_cards: bool,
) -> int:
    """Higher = better to drink when HP is critical."""
    if profile.passive or profile.self_damage:
        return -1000
    score = 0
    if profile.heal:
        score += 100
    if profile.block:
        score += 80 + min(gap, 30)
    if profile.defensive:
        score += 75 + min(gap // 2, 20)
    if profile.debuff:
        score += 55 + min(gap // 2, 25)
    if profile.draw:
        score += 60 if not has_playable_cards else 45
    if profile.buff:
        score += 40
    if profile.offensive:
        score += 25
    return score


def score_potion(
    potion: dict | None,
    *,
    hp_ratio: float = 1.0,
    kb: KnowledgeBase | None = None,
) -> float:
    """Higher = keep this potion over others."""
    if not potion or not isinstance(potion, dict):
        return 0.0
    kb = kb or get_knowledge()
    profile = get_potion_profile(potion, kb)

    score = 45.0
    if profile.heal:
        score = 72.0 if hp_ratio < 0.55 else 48.0
    elif profile.offensive or profile.debuff:
        score = 68.0
    elif profile.block or profile.defensive:
        score = 62.0
    elif profile.buff or profile.draw:
        score = 58.0
    elif profile.passive:
        score = 50.0

    if potion.get("can_use_in_combat") is False:
        score -= 15
    if profile.self_damage:
        score -= 20

    if kb.lookup_potion(potion_lookup_key(potion)):
        score += 5

    return score


def score_offered_potion_reward(item: dict, state: dict, kb: KnowledgeBase | None = None) -> float:
    player = state.get("player") or {}
    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    hp_ratio = hp / max_hp if max_hp else 1.0
    offered = {
        "name": item.get("potion_name") or item.get("name"),
        "id": item.get("potion_id") or item.get("id"),
        "description": item.get("description") or item.get("potion_description"),
    }
    return score_potion(offered, hp_ratio=hp_ratio, kb=kb)


def worst_potion_slot(player: dict, kb: KnowledgeBase | None = None) -> tuple[int, float]:
    """Slot index and score of the weakest potion to discard."""
    kb = kb or get_knowledge()
    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    hp_ratio = hp / max_hp if max_hp else 1.0

    slots = filled_potion_slots(player)
    if not slots:
        return 0, 0.0

    worst_slot, worst_potion = min(
        slots,
        key=lambda pair: score_potion(pair[1], hp_ratio=hp_ratio, kb=kb),
    )
    return worst_slot, score_potion(worst_potion, hp_ratio=hp_ratio, kb=kb)

