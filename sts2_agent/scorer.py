"""Scoring functions for combat, rewards, map, and rest - backed by KnowledgeBase."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sts2_agent.knowledge import KnowledgeBase, _normalize_name

RARITY_BONUS = {"common": 0, "uncommon": 12, "rare": 25, "starter": 0, "basic": 0, "curse": -100}
SCALING_KEYWORDS = frozenset(
    {
        "strength",
        "dexterity",
        "focus",
        "vulnerable",
        "weak",
        "thorns",
        "metallicize",
        "plated",
        "ritual",
        "demon",
        "echo",
        "loop",
        "inflame",
    }
)
DEBUFF_POWERS = frozenset({"vulnerable", "weak", "frail", "poison", "daze", "hex", "doom"})
NON_ATTACK_INTENT_LABELS = frozenset(
    {
        "empower",
        "enrage",
        "strengthen",
        "heal",
        "buff",
        "strategic",
        "skulk",
        "curl",
        "defend",
        "defensive",
        "block",
        "sleep",
        "stun",
    }
)
SMITH_PRIORITY = (
    ("bash", 100),
    ("strike", 80),
    ("defend", 75),
)


@dataclass
class ScoredOption:
    score: float
    reasons: list[str] = field(default_factory=list)
    index: int | None = None
    label: str = ""


def card_name(card: dict | str) -> str:
    if isinstance(card, str):
        return card
    return str(card.get("name") or card.get("id") or "")


def deck_cards(state: dict) -> list[dict]:
    player = state.get("player") or {}
    piles = []
    for key in ("deck", "draw_pile", "discard_pile", "hand", "exhaust_pile"):
        pile = player.get(key)
        if isinstance(pile, list):
            piles.extend(pile)
    return [c for c in piles if isinstance(c, dict)]


def deck_size(state: dict) -> int:
    player = state.get("player") or {}
    if isinstance(player.get("deck"), list):
        return len(player["deck"])
    return len(deck_cards(state))


def deck_keywords(deck: list[dict], kb: KnowledgeBase) -> set[str]:
    keywords: set[str] = set()
    for card in deck:
        name = card_name(card)
        codex = kb.lookup_card(name)
        if not codex:
            continue
        for kw in codex.get("keywords") or []:
            keywords.add(str(kw).lower())
        for kw in codex.get("keywords_key") or []:
            keywords.add(str(kw).lower())
        for power in codex.get("powers_applied") or []:
            if isinstance(power, dict) and power.get("power"):
                keywords.add(str(power["power"]).lower())
    return keywords


def score_card_reward(
    card: dict | str,
    state: dict,
    kb: KnowledgeBase,
) -> ScoredOption:
    """Score a card offer for reward selection."""
    name = card_name(card)
    norm = _normalize_name(name)
    reasons: list[str] = []
    score = 10.0

    codex = kb.lookup_card(name)
    if codex:
        score, reasons = _score_from_codex(codex, score, reasons)
    elif isinstance(card, dict):
        score += _rarity_bonus(str(card.get("rarity") or ""), reasons)

    run = state.get("run") or {}
    floor = int(run.get("floor") or 0)
    if floor < 20:
        if codex and (codex.get("damage") or _is_attack_type(codex)):
            score += 25
            reasons.append("early run: favor damage")
    elif floor < 40:
        if codex and _has_scaling(codex):
            score += 22
            reasons.append("mid run: favor scaling")
    else:
        score += 10
        reasons.append("late run: favor consistency")

    win_rate = kb.community_win_rate(name)
    if win_rate is not None:
        community_pts = (win_rate - 0.5) * 80
        score += community_pts
        reasons.append(f"community win rate {win_rate:.1%} ({community_pts:+.0f})")
    else:
        reasons.append("no community win rate (using Codex heuristics)")

    pick_rate = kb.community_pick_rate(name)
    if pick_rate is not None:
        score += pick_rate * 20
        reasons.append(f"community pick rate {pick_rate:.1%}")

    expert_bonus, expert_tier = kb.expert_card_bonus(name)
    if expert_bonus:
        score += expert_bonus
        reasons.append(f"Mobalytics tier {expert_tier} ({expert_bonus:+.0f})")
    expert_notes = kb.expert_card_notes(name)
    if expert_notes:
        reasons.append(expert_notes[:120])

    player = state.get("player") or {}
    character = str(player.get("character") or run.get("character") or "").lower()
    archetypes = kb.character_archetypes(character)
    if archetypes and codex:
        arch_text = " ".join(archetypes).lower()
        codex_text = " ".join(
            [
                str(codex.get("name") or ""),
                str(codex.get("description") or ""),
                " ".join(str(k) for k in codex.get("keywords") or []),
            ]
        ).lower()
        if any(token in codex_text for token in arch_text.split() if len(token) > 4):
            score += 12
            reasons.append(f"fits {character or 'deck'} archetype")

    dk = deck_keywords(deck_cards(state), kb)
    if codex:
        synergy = _synergy_bonus(codex, dk)
        if synergy:
            score += synergy
            reasons.append(f"deck synergy +{synergy:.0f}")

    size = deck_size(state)
    if size > 20:
        penalty = (size - 20) * 2
        score -= penalty
        reasons.append(f"deck size {size} penalty -{penalty:.0f}")

    if any(c in norm for c in ("curse", "injury", "pain", "regret", "void")):
        score = -100
        reasons.append("curse/status - skip")

    return ScoredOption(score=score, reasons=reasons, label=name)


def _score_from_codex(codex: dict, score: float, reasons: list[str]) -> tuple[float, list[str]]:
    rarity = str(codex.get("rarity_key") or codex.get("rarity") or "").lower()
    score += _rarity_bonus(rarity, reasons)

    cost = codex.get("cost")
    if cost is not None and not codex.get("is_x_cost"):
        cost = max(int(cost), 0)
        damage = int(codex.get("damage") or 0)
        block = int(codex.get("block") or 0)
        hits = int(codex.get("hit_count") or 1)
        if damage and cost >= 0:
            efficiency = (damage * hits) / max(cost, 1)
            pts = efficiency * 4
            score += pts
            reasons.append(f"damage/energy efficiency {efficiency:.1f} (+{pts:.0f})")
        if block and cost >= 0:
            efficiency = block / max(cost, 1)
            pts = efficiency * 2.5
            score += pts
            reasons.append(f"block/energy efficiency {efficiency:.1f} (+{pts:.0f})")

    if _has_scaling(codex):
        score += 18
        reasons.append("scaling card +18")
    elif _is_attack_type(codex):
        score += 12
        reasons.append("attack card +12")
    elif str(codex.get("type_key") or "").lower() == "skill" and codex.get("block"):
        score += 8
        reasons.append("block skill +8")

    card_type = str(codex.get("type_key") or codex.get("type") or "").lower()
    if card_type == "power":
        score += 15
        reasons.append("power card +15")

    return score, reasons


def _rarity_bonus(rarity: str, reasons: list[str]) -> float:
    rarity = rarity.lower().replace(" relic", "")
    for key, bonus in RARITY_BONUS.items():
        if key in rarity:
            if bonus:
                reasons.append(f"rarity {key} +{bonus}")
            return float(bonus)
    return 0.0


def _is_attack_type(codex: dict) -> bool:
    t = str(codex.get("type_key") or codex.get("type") or "").lower()
    return "attack" in t or bool(codex.get("damage"))


def _has_scaling(codex: dict) -> bool:
    text = " ".join(
        [
            str(codex.get("description") or ""),
            " ".join(str(k) for k in (codex.get("keywords") or [])),
        ]
    ).lower()
    if any(k in text for k in SCALING_KEYWORDS):
        return True
    for power in codex.get("powers_applied") or []:
        if isinstance(power, dict):
            pname = str(power.get("power") or "").lower()
            if pname in SCALING_KEYWORDS or pname in DEBUFF_POWERS:
                return True
    return False


def _synergy_bonus(codex: dict, deck_keywords: set[str]) -> float:
    bonus = 0.0
    card_kw = {str(k).lower() for k in (codex.get("keywords") or [])}
    card_kw |= {str(k).lower() for k in (codex.get("keywords_key") or [])}
    overlap = card_kw & deck_keywords
    if overlap:
        bonus += 8 * len(overlap)
    for power in codex.get("powers_applied") or []:
        if isinstance(power, dict):
            pname = str(power.get("power") or "").lower()
            if pname in deck_keywords:
                bonus += 10
    return bonus


def _intent_api_label(intent: dict) -> str:
    for key in ("title", "name", "label", "intent", "api_label"):
        val = intent.get(key)
        if val:
            return str(val).strip().lower()
    return ""


def intent_is_attack(intent: Any) -> bool:
    """True only when the intent is an attack/damage move (not defend/buff/debuff)."""
    if isinstance(intent, str):
        text = intent.lower()
        if any(k in text for k in ("defend", "block", "buff", "strength", "ritual", "empower")):
            return False
        return "attack" in text or "damage" in text
    if not isinstance(intent, dict):
        return False
    api_label = _intent_api_label(intent)
    if api_label in NON_ATTACK_INTENT_LABELS:
        return False
    itype = str(
        intent.get("type") or intent.get("intent_type") or intent.get("intent") or ""
    ).lower()
    if not itype:
        itype = str(intent.get("text") or intent.get("description") or "").lower()
    if itype in NON_ATTACK_INTENT_LABELS or any(
        k in itype for k in ("defend", "block", "buff", "debuff", "strength", "ritual", "empower")
    ):
        return False
    if "attack" in itype or "damage" in itype:
        return _intent_damage_value(intent) > 0 or api_label in ("aggressive", "attack")
    text = str(intent.get("text") or intent.get("description") or "").lower()
    if any(k in text for k in ("defend", "gain", " block", "buff", "strength", "empower")):
        if "attack" not in text and "damage" not in text:
            return False
    if "attack" in text or "damage" in text:
        return _intent_damage_value(intent) > 0 or api_label in ("aggressive", "attack")
    return False


def card_applies_vulnerable(card: dict, kb: KnowledgeBase | None = None) -> bool:
    """True when playing this card applies Vulnerable to an enemy."""
    kb = kb or get_knowledge()
    name = card_name(card).lower()
    if "bash" in name:
        return True
    codex = kb.lookup_card(card_name(card))
    if codex:
        for power in codex.get("powers_applied") or []:
            if isinstance(power, dict):
                pname = str(power.get("power") or "").lower()
                if "vulnerable" in pname:
                    return True
        desc = str(codex.get("description") or codex.get("description_raw") or "").lower()
        if "vulnerable" in desc and "apply" in desc:
            return True
    desc = str(card.get("description") or "").lower()
    return "vulnerable" in desc


def intent_threat_score(intent: Any) -> tuple[float, str]:
    """Return (threat_score, label) for a single enemy intent."""
    if isinstance(intent, str):
        itype = intent.lower()
        damage = _parse_damage_text(intent)
    elif isinstance(intent, dict):
        itype = str(intent.get("type") or intent.get("intent_type") or intent.get("intent") or "").lower()
        damage = _intent_damage_value(intent)
        if not itype:
            itype = str(intent.get("text") or intent.get("description") or "").lower()
    else:
        return 1.0, "unknown"

    if "attack" in itype or "damage" in itype:
        dmg = damage or 8
        return float(dmg), f"attack ~{dmg}"
    if "debuff" in itype or any(k in itype for k in DEBUFF_POWERS):
        return 50.0, "debuff"
    if "buff" in itype or "strength" in itype or "ritual" in itype:
        return 30.0, "buff"
    if "block" in itype or "defend" in itype:
        return 5.0, "defensive"
    if "unknown" in itype or "none" in itype or not itype:
        return 3.0, "unknown"
    return 10.0, itype


def enemy_total_threat(enemy: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    total = 0.0
    intents = enemy.get("intents") or []
    if isinstance(enemy.get("intent"), dict):
        intents = [enemy["intent"]]
    elif isinstance(enemy.get("intent"), list):
        intents = enemy["intent"]
    for intent in intents:
        threat, label = intent_threat_score(intent)
        total += threat
        reasons.append(f"{enemy.get('name', '?')}: {label} ({threat:.0f})")
    if not intents:
        total = 3.0
        reasons.append(f"{enemy.get('name', '?')}: no intent (low)")
    return total, reasons


def total_incoming_attack_damage(enemies: list[dict]) -> tuple[int, list[str]]:
    total = 0
    reasons: list[str] = []
    for enemy in enemies:
        if int(enemy.get("hp") or 0) <= 0:
            continue
        intents = enemy.get("intents") or []
        if isinstance(enemy.get("intent"), dict):
            intents = [enemy["intent"]]
        for intent in intents:
            if not intent_is_attack(intent):
                _, label = intent_threat_score(intent)
                reasons.append(
                    f"{enemy.get('name')}: {label} - not counted as incoming"
                )
                continue
            threat, label = intent_threat_score(intent)
            dmg = _intent_damage_value(intent)
            if dmg <= 0:
                dmg = int(threat) if threat > 5 and label.startswith("attack") else 0
            if dmg <= 0:
                reasons.append(
                    f"{enemy.get('name')}: {label} dmg=0 - not counted as incoming"
                )
                continue
            total += dmg
            reasons.append(f"incoming attack from {enemy.get('name')}: {dmg} ({label})")
    if total == 0 and enemies:
        reasons.append("incoming attacks this turn: 0")
    return total, reasons


def pick_highest_threat_enemy(enemies: list[dict]) -> tuple[dict | None, list[str]]:
    living = [e for e in enemies if int(e.get("hp") or 0) > 0]
    if not living:
        return None, ["no living enemies"]
    best = max(living, key=lambda e: enemy_total_threat(e)[0])
    threat, reasons = enemy_total_threat(best)
    return best, [f"target highest threat: {best.get('name')} (score {threat:.0f})"] + reasons


def enemy_has_vulnerable(enemy: dict) -> bool:
    status = enemy.get("status") or enemy.get("powers") or []
    if isinstance(status, dict):
        status = list(status.values())
    for s in status:
        if isinstance(s, dict):
            name = str(s.get("id") or s.get("name") or s.get("power") or "").lower()
        else:
            name = str(s).lower()
        if "vulnerable" in name:
            return True
    return False


def estimate_hand_damage(
    hand: list[dict],
    energy: int,
    enemies: list[dict],
    kb: KnowledgeBase,
) -> tuple[int, list[str]]:
    """Greedy upper bound on attack damage playable this turn."""
    living = [e for e in enemies if int(e.get("hp") or 0) > 0]
    vuln_mult = 1.5 if any(enemy_has_vulnerable(e) for e in living) else 1.0
    options: list[tuple[float, int]] = []

    for card in hand:
        if card.get("can_play") is False or card.get("playable") is False:
            continue
        cost = _card_cost(card)
        if cost > energy:
            continue
        codex = kb.lookup_card(card_name(card))
        if not _hand_card_is_attack(card, codex):
            continue
        dmg = _card_damage_value(card, codex) * vuln_mult
        options.append((dmg, cost))

    options.sort(key=lambda x: (-x[0], x[1]))
    total = 0
    remaining = energy
    for dmg, cost in options:
        if cost <= remaining:
            total += int(dmg)
            remaining -= cost

    reasons = [f"estimated hand damage: {total} (energy {energy})"]
    return total, reasons


def find_lethal_target(
    enemies: list[dict],
    damage: int,
) -> tuple[dict | None, list[str]]:
    """Pick a killable enemy; prefer high incoming threat over chip damage."""
    living = [e for e in enemies if int(e.get("hp") or 0) > 0]
    killable = [e for e in living if int(e.get("hp") or 0) <= damage]
    if not killable:
        return None, []

    def _priority(enemy: dict) -> tuple[float, int]:
        threat, _ = enemy_total_threat(enemy)
        return (threat, -int(enemy.get("hp") or 0))

    target = max(killable, key=_priority)
    return target, [
        f"lethal: {target.get('name')} has {target.get('hp')} HP, "
        f"hand can deal ~{damage}"
    ]


def score_combat_play(
    card: dict,
    *,
    energy: int,
    incoming_damage: int,
    current_block: int,
    enemies: list[dict],
    kb: KnowledgeBase,
    floor: int,
    lethal_damage: int | None = None,
    next_turn_incoming: int = 0,
    expected_block_next_turn: int = 0,
) -> ScoredOption:
    """Score playing a card this turn (higher = play sooner)."""
    name = card_name(card)
    reasons: list[str] = [f"evaluating {name}"]
    score = 0.0

    if card.get("can_play") is False or card.get("playable") is False:
        return ScoredOption(0, ["not playable"], label=name)

    cost = _card_cost(card)
    if cost > energy:
        return ScoredOption(0, [f"cost {cost} > energy {energy}"], label=name)

    codex = kb.lookup_card(name)
    no_incoming_attacks = incoming_damage <= 0
    needs_block = incoming_damage > current_block
    gap = incoming_damage - current_block
    safe = not needs_block

    is_block = _hand_card_is_block(card, codex)
    is_attack = _hand_card_is_attack(card, codex)
    is_debuff = _hand_card_is_debuff(card, codex)
    is_power = _hand_card_is_power(card, codex)
    pure_block = is_block and not is_attack

    block_val = _card_block_value(card, codex)
    damage_val = _card_damage_value(card, codex)

    living = [e for e in enemies if int(e.get("hp") or 0) > 0]
    min_hp = min((int(e.get("hp") or 0) for e in living), default=999)
    lethal_this_card = is_attack and damage_val >= min_hp
    lethal_turn = lethal_damage is not None and lethal_damage >= min_hp

    if lethal_turn and is_attack:
        score += 200 + damage_val * 4
        reasons.append(f"lethal turn - kill priority (+{200 + damage_val * 4:.0f})")
    elif lethal_this_card and is_attack:
        score += 150 + damage_val * 3
        reasons.append("this card can kill weakest enemy")

    if needs_block and pure_block and not lethal_turn:
        score += 120 + block_val * 5
        reasons.append(f"needs block (gap {gap}): block card +{block_val * 5}")
    elif pure_block and not needs_block:
        if no_incoming_attacks:
            score += 5 + min(block_val, 5)
            reasons.append("pure block - 0 incoming attacks (very low priority)")
        else:
            score += 12 + block_val
            reasons.append("pure block (low priority)")
        next_risk = max(next_turn_incoming - expected_block_next_turn, 0)
        if next_risk > 8 and not no_incoming_attacks:
            score += 25 + min(next_risk, 20)
            reasons.append(f"next-turn risk {next_risk} - keep block (+{25 + min(next_risk, 20)})")
        elif next_risk > 8 and no_incoming_attacks:
            score += min(8, next_risk)
            reasons.append(f"next-turn risk {next_risk} - small block bias (+{min(8, next_risk)})")
    elif is_block and is_attack:
        if no_incoming_attacks:
            score += 55 + damage_val * 3 + min(block_val, 5)
            reasons.append("block+attack hybrid - favor damage (0 incoming)")
        else:
            score += 40 + damage_val * 2 + block_val
            reasons.append("block+attack hybrid")

    if is_debuff and not lethal_turn:
        score += 75
        reasons.append("debuff before damage (+75)")
        if any(enemy_total_threat(e)[0] >= 30 for e in living):
            score += 20
            reasons.append("high-threat enemy present (+20)")

    if card_applies_vulnerable(card, kb) and not any(enemy_has_vulnerable(e) for e in living):
        score += 130
        reasons.append("apply Vulnerable before attacks (+130)")

    if is_attack and not lethal_turn:
        base = 90 + damage_val * 3
        score += base
        reasons.append(f"attack +{base:.0f}")
        if not any(enemy_has_vulnerable(e) for e in living) and not card_applies_vulnerable(
            card, kb
        ):
            score -= 50
            reasons.append("enemy not Vulnerable - attack deprioritized (-50)")
        if no_incoming_attacks:
            score += 40
            reasons.append("0 incoming attacks - press damage +40")
        if any(enemy_has_vulnerable(e) for e in living):
            score += 25
            reasons.append("vulnerable enemy bonus +25")
        next_risk = max(next_turn_incoming - expected_block_next_turn, 0)
        if safe and next_risk <= 5:
            score += 20
            reasons.append("safe now + low next-turn risk - press attack +20")
        elif safe and next_risk > 12 and not no_incoming_attacks:
            score -= 15
            reasons.append("safe now but risky next turn - attack -15")

    if is_power:
        if safe or incoming_damage == 0:
            score += 95 if floor <= 15 else 80
            reasons.append("power - safe to setup")
        elif current_block >= incoming_damage:
            score += 70
            reasons.append("power - already enough block")
        else:
            score += 20
            reasons.append("power - deprioritized (must survive)")

    if cost == 0:
        score += 15
        reasons.append("zero cost +15")

    score -= cost * 2
    return ScoredOption(score=score, reasons=reasons, label=name)


def affordable_block_total(hand: list[dict], energy: int, kb: KnowledgeBase) -> int:
    total = 0
    remaining = energy
    block_cards: list[tuple[int, int]] = []
    for card in hand:
        if card.get("can_play") is False:
            continue
        cost = _card_cost(card)
        if cost > remaining:
            continue
        codex = kb.lookup_card(card_name(card))
        if _hand_card_is_block(card, codex):
            block_cards.append((cost, _card_block_value(card, codex)))
    block_cards.sort(key=lambda x: (-x[1], x[0]))
    for cost, block in block_cards:
        if cost <= remaining:
            total += block
            remaining -= cost
    return total


def score_map_room(
    room: str,
    *,
    hp_ratio: float,
    gold: int,
    boss_soon: bool,
) -> ScoredOption:
    room = room.replace(" ", "_").lower()
    reasons: list[str] = [f"room type: {room}"]

    if _room_is(room, "treasure"):
        return ScoredOption(95, reasons + ["treasure always high"], label=room)
    if _room_is(room, "rest"):
        score = 40
        if hp_ratio < 0.5:
            score = 98
            reasons.append("HP < 50% - rest critical")
        elif hp_ratio < 0.8:
            score = 75
            reasons.append("HP < 80% - rest valuable")
        else:
            reasons.append("HP high - rest low value")
        if boss_soon:
            score += 25
            reasons.append("boss soon +25")
        return ScoredOption(score, reasons, label=room)
    if _room_is(room, "elite"):
        if hp_ratio > 0.6:
            score = 85
            reasons.append("HP > 60% - elite high value")
        elif hp_ratio < 0.4:
            score = 15
            reasons.append("HP < 40% - elite dangerous")
        else:
            score = 50
            reasons.append("HP moderate - elite medium")
        return ScoredOption(score, reasons, label=room)
    if _room_is(room, "event"):
        return ScoredOption(60, reasons + ["event medium (unpredictable)"], label=room)
    if _room_is(room, "shop"):
        score = 35
        if gold > 150:
            score = 80
            reasons.append(f"gold {gold} > 150 - shop high value")
        return ScoredOption(score, reasons, label=room)
    if _room_is(room, "monster"):
        score = 25
        if hp_ratio < 0.45:
            score -= 15
            reasons.append("low HP - avoid monsters")
        reasons.append("monster low value (risk, weak reward)")
        return ScoredOption(score, reasons, label=room)
    if _room_is(room, "boss"):
        return ScoredOption(5, reasons + ["boss node"], label=room)
    return ScoredOption(45, reasons + ["unknown room default"], label=room)


def score_shop_item(
    item: dict,
    *,
    state: dict,
    hp_ratio: float,
    gold: int,
    owned_relic_ids: set[str],
    owned_relic_names: set[str],
    kb: KnowledgeBase,
) -> ScoredOption:
    from sts2_agent.state_parse import shop_item_category, shop_item_name, shop_item_price

    category = shop_item_category(item)
    name = shop_item_name(item)
    price = shop_item_price(item)
    label = f"{category}:{name}" if name else category
    reasons = [f"{label} @ {price}g"]

    if item.get("on_sale"):
        reasons.append("on sale")

    if category == "card_removal":
        size = deck_size(state)
        if size > 18:
            return ScoredOption(88, reasons + [f"deck size {size} - removal valuable"], label=label)
        if size > 14:
            return ScoredOption(65, reasons + [f"deck size {size} - removal ok"], label=label)
        return ScoredOption(25, reasons + ["deck small - skip removal"], label=label)

    if category == "relic":
        relic_id = str(item.get("relic_id") or "").lower()
        norm_name = _normalize_name(name)
        if relic_id and relic_id in owned_relic_ids:
            return ScoredOption(0, reasons + ["already own relic"], label=label)
        if norm_name and norm_name in owned_relic_names:
            return ScoredOption(0, reasons + ["already own relic"], label=label)
        score = 110.0
        codex = kb.lookup_relic(name) if name else None
        if codex:
            score += 10
            reasons.append("known relic +10")
        return ScoredOption(score, reasons, label=label)

    if category == "potion":
        potion_slots = int((state.get("player") or {}).get("max_potion_slots") or 3)
        potions = (state.get("player") or {}).get("potions") or []
        filled = sum(1 for p in potions if p)
        if hp_ratio < 0.55:
            score = 72.0
            reasons.append("low HP - potion useful")
        elif filled < potion_slots:
            score = 48.0
            reasons.append(f"potion slot free ({filled}/{potion_slots})")
        else:
            score = 18.0
            reasons.append("potion slots full - low priority")
        return ScoredOption(score, reasons, label=label)

    if category == "card":
        card_payload = {
            "name": name,
            "rarity": item.get("card_rarity") or item.get("rarity"),
            "type": item.get("card_type") or item.get("type"),
            "description": item.get("card_description") or item.get("description"),
        }
        picked = score_card_reward(card_payload, state, kb)
        score = picked.score
        reasons.extend(picked.reasons)
        if price > 0 and price <= gold:
            value_ratio = score / max(price / 25.0, 1.0)
            score += min(value_ratio * 5, 15)
            reasons.append(f"price/value adj +{min(value_ratio * 5, 15):.0f}")
        return ScoredOption(score, reasons, label=label)

    return ScoredOption(20, reasons + ["unknown shop item"], label=label)


def score_rest_option(
    option: dict | str,
    *,
    hp_ratio: float,
    deck: list[dict],
    kb: KnowledgeBase,
) -> ScoredOption:
    label = _option_label(option)
    reasons = [f"rest option: {label}"]

    if any(k in label for k in ("rest", "heal", "sleep")):
        if hp_ratio < 0.5:
            return ScoredOption(100, reasons + ["HP < 50% - always rest"], label=label)
        if hp_ratio > 0.8:
            return ScoredOption(25, reasons + ["HP > 80% - rest low value"], label=label)
        return ScoredOption(70, reasons + ["moderate HP - rest ok"], label=label)

    if any(k in label for k in ("smith", "upgrade")):
        upgrade_score = _best_upgrade_in_deck(deck, kb)
        if hp_ratio > 0.7 and upgrade_score >= 70:
            return ScoredOption(
                90,
                reasons + [f"HP > 70% and strong upgrade available (prio {upgrade_score})"],
                label=label,
            )
        if hp_ratio > 0.5:
            return ScoredOption(55, reasons + ["smith when healthy"], label=label)
        return ScoredOption(30, reasons + ["smith while hurt - risky"], label=label)

    return ScoredOption(40, reasons + ["default rest option"], label=label)


def card_remove_priority(card: dict | str, kb: KnowledgeBase) -> ScoredOption:
    """Higher score = better candidate to remove (strikes/defends before rares)."""
    name = card_name(card)
    norm = _normalize_name(name)
    reasons = [f"remove candidate: {name}"]
    score = 25.0

    if "strike" in norm:
        score = 85.0
        reasons.append("basic attack - good remove")
    elif "defend" in norm:
        score = 75.0
        reasons.append("basic skill - ok remove")
    elif norm.startswith("BASH"):
        score = 15.0
        reasons.append("keep core card")

    codex = kb.lookup_card(name)
    if codex:
        rarity = str(codex.get("rarity_key") or "").lower()
        if "rare" in rarity:
            score = min(score, 10.0)
            reasons.append("rare - avoid removing")
        elif "uncommon" in rarity:
            score = min(score, 35.0)

    return ScoredOption(score=score, reasons=reasons, label=name)


def smith_upgrade_priority(card: dict | str, kb: KnowledgeBase) -> ScoredOption:
    name = card_name(card)
    norm = _normalize_name(name)
    reasons = [f"upgrade candidate: {name}"]
    score = 20.0

    for prefix, pts in SMITH_PRIORITY:
        if norm.startswith(prefix):
            score = float(pts)
            reasons.append(f"priority upgrade ({prefix}) +{pts}")
            break

    codex = kb.lookup_card(name)
    if codex:
        rarity = str(codex.get("rarity_key") or "").lower()
        if "rare" in rarity:
            score += 40
            reasons.append("rare card +40")
        elif "uncommon" in rarity and _is_attack_type(codex):
            score += 25
            reasons.append("uncommon attack +25")

    return ScoredOption(score=score, reasons=reasons, label=name)


def _best_upgrade_in_deck(deck: list[dict], kb: KnowledgeBase) -> float:
    if not deck:
        return 0.0
    return max(smith_upgrade_priority(c, kb).score for c in deck)


def _room_is(room: str, kind: str) -> bool:
    normalized = room.lower().replace("_", "").replace(" ", "")
    keys = {
        "elite": ("elite",),
        "rest": ("rest", "restsite", "rest_site"),
        "event": ("event", "?", "question", "unknown"),
        "monster": ("monster", "enemy", "combat", "battle"),
        "shop": ("shop", "merchant"),
        "treasure": ("treasure", "chest"),
        "boss": ("boss",),
        "ancient": ("ancient",),
    }
    return any(k in normalized for k in keys.get(kind, ()))


def _option_label(option: dict | str) -> str:
    if isinstance(option, str):
        return option.lower()
    return str(
        option.get("id")
        or option.get("label")
        or option.get("name")
        or option.get("type")
        or ""
    ).lower()


def _card_cost(card: dict) -> int:
    for key in ("cost", "energy_cost", "energy"):
        val = card.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            text = val.strip().upper()
            if text in ("X", ""):
                return 99
            try:
                return int(text)
            except ValueError:
                continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return 0


def _hand_card_is_block(card: dict, codex: dict | None) -> bool:
    if card.get("block") or card.get("block_gain"):
        return True
    if codex and codex.get("block"):
        return True
    name = card_name(card).lower()
    return any(k in name for k in ("defend", "block", "arm", "ward"))


def _hand_card_is_attack(card: dict, codex: dict | None) -> bool:
    if codex and _is_attack_type(codex):
        return True
    return "attack" in str(card.get("type") or "").lower()


def _hand_card_is_debuff(card: dict, codex: dict | None) -> bool:
    name = card_name(card).lower()
    if any(k in name for k in DEBUFF_POWERS):
        return True
    if codex:
        for power in codex.get("powers_applied") or []:
            if isinstance(power, dict):
                pname = str(power.get("power") or "").lower()
                if pname in DEBUFF_POWERS:
                    return True
    return False


def _hand_card_is_power(card: dict, codex: dict | None) -> bool:
    if codex:
        return str(codex.get("type_key") or "").lower() == "power"
    return "power" in str(card.get("type") or "").lower()


def _card_block_value(card: dict, codex: dict | None) -> int:
    for source in (card, codex or {}):
        if source.get("block") is not None:
            return int(source["block"])
    return 5 if _hand_card_is_block(card, codex) else 0


def _card_damage_value(card: dict, codex: dict | None) -> int:
    for source in (card, codex or {}):
        if source.get("damage") is not None:
            dmg = int(source["damage"])
            hits = int(source.get("hit_count") or 1)
            return dmg * hits
    for source in (card, codex or {}):
        desc = str(source.get("description") or "")
        parsed = _parse_damage_text(desc)
        if parsed:
            return parsed
    return 6 if _hand_card_is_attack(card, codex) else 0


def _intent_damage_value(intent: dict) -> int:
    for key in ("damage", "min_damage", "max_damage", "value"):
        if intent.get(key) is not None:
            try:
                return int(intent[key])
            except (TypeError, ValueError):
                pass
    text = str(intent.get("text") or intent.get("description") or "")
    return _parse_damage_text(text)


def _parse_damage_text(text: str) -> int:
    match = re.search(r"(\d+)\s*(?:damage|dmg)", text, re.I)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d{1,3})\b", text)
    if match:
        return int(match.group(1))
    return 0


def combat_reward(hp_before: int, hp_after: int, max_hp: int, won_combat: bool) -> float:
    """Per-combat score from HP conservation (higher is better)."""
    max_hp = max(int(max_hp), 1)
    if not won_combat:
        return -100.0

    hp_lost = max(0, int(hp_before) - int(hp_after))
    hp_lost_pct = hp_lost / max_hp

    if hp_lost_pct == 0:
        return 30.0
    if hp_lost_pct <= 0.10:
        return 15.0
    if hp_lost_pct <= 0.25:
        return 5.0
    if hp_lost_pct <= 0.40:
        return -10.0
    if hp_lost_pct <= 0.60:
        return -25.0
    return -50.0


DECK_TIER_VALUES: dict[str, int] = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
DECK_QUALITY_MULTIPLIER = 25.0  # ~50-75 pts for a strong 20-card deck; below floor_score at 12 floors


def _is_starter_card(card_id: str) -> bool:
    """Basic strikes/defends still in the final deck indicate weak deck building."""
    upper = str(card_id).upper().replace(" ", "_")
    if upper in ("STRIKE", "DEFEND", "SETUP_STRIKE"):
        return True
    if upper.endswith("_STRIKE") or upper.endswith("_DEFEND"):
        # STRIKE_IRONCLAD, IRONCLAD_STRIKE — not PERFECTED_STRIKE (no _STRIKE suffix alone)
        parts = upper.split("_")
        if len(parts) == 2 and parts[-1] in ("STRIKE", "DEFEND"):
            return True
    return False


def _card_tier_value(card_id: str, kb: KnowledgeBase) -> int:
    if _is_starter_card(card_id):
        return DECK_TIER_VALUES["D"]
    tier = kb.expert_card_tier(card_id)
    if tier:
        letter = str(tier).strip().upper()[:1]
        return DECK_TIER_VALUES.get(letter, DECK_TIER_VALUES["C"])
    return DECK_TIER_VALUES["C"]


def deck_quality_score(final_deck: object, kb: KnowledgeBase | None = None) -> float:
    """Mean expert tier value of final_deck * multiplier (Mobalytics tiers in expert_knowledge.json)."""
    if not isinstance(final_deck, list) or not final_deck:
        return 0.0

    if kb is None:
        from sts2_agent.knowledge import get_knowledge

        kb = get_knowledge()

    values: list[int] = []
    for card in final_deck:
        if isinstance(card, dict):
            card = card.get("name") or card.get("id")
        if not card:
            continue
        values.append(_card_tier_value(str(card), kb))

    if not values:
        return 0.0

    mean_tier = sum(values) / len(values)
    return mean_tier * DECK_QUALITY_MULTIPLIER


def damage_efficiency_penalty(combat_summary: object) -> float:
    """Penalize low damage-per-turn across fights (Phase B combat_summary only).

    Balanced play is ~10+ damage/turn; passive block-heavy runs sit at 5-8.
    Threshold 10, multiplier -12: max penalty -120 at 0 dpt, zero at 10+ dpt.
    """
    if not isinstance(combat_summary, list) or not combat_summary:
        return 0.0

    total_turns = 0
    total_damage = 0
    for fight in combat_summary:
        if not isinstance(fight, dict):
            continue
        total_turns += int(fight.get("turns") or 0)
        total_damage += int(fight.get("damage_dealt") or 0)

    if total_turns <= 0:
        return 0.0

    damage_per_turn = total_damage / total_turns
    return max(0.0, 10.0 - damage_per_turn) * -12.0


def run_score(run_data: dict[str, Any]) -> float:
    """Total run score from progression, HP conservation, and outcome.

    floors*15 + (act-1)*60 + avg_hp_pct*50 + deck_quality*25 + win_bonus(1000)
    + bosses*100 + damage_efficiency_penalty (Phase B combat_summary).
    """
    floors = int(run_data.get("floors_reached") or 0)
    act = int(run_data.get("act_reached") or 1)
    avg_hp_remaining = float(run_data.get("avg_hp_pct_after_combat") or 0.0)
    bosses_killed = int(run_data.get("bosses_killed") or 0)
    won = bool(run_data.get("won"))

    floor_score = floors * 15
    act_bonus = (act - 1) * 60
    hp_conservation_score = avg_hp_remaining * 50.0
    win_bonus = 1000.0 if won else 0.0
    boss_bonus = bosses_killed * 100.0
    deck_score = deck_quality_score(run_data.get("final_deck"))
    dmg_penalty = damage_efficiency_penalty(run_data.get("combat_summary"))

    total = (
        floor_score
        + act_bonus
        + hp_conservation_score
        + deck_score
        + win_bonus
        + boss_bonus
        + dmg_penalty
    )
    return max(0.0, float(total))


def combat_turn_shaping(
    hp_lost_this_turn: int,
    block_applied: int,
    damage_dealt: int,
) -> float:
    """Small per-action signal during combat (before end-of-combat payout)."""
    return (
        -float(hp_lost_this_turn) * 2.0
        + float(block_applied) * 0.15
        + float(damage_dealt) * 0.25
    )
