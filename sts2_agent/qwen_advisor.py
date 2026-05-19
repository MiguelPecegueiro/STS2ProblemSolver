"""Per-fight combat strategy via local Qwen (LM Studio) API."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from sts2_agent.knowledge import EXPERT_KNOWLEDGE_PATH, _normalize_name, get_knowledge

logger = logging.getLogger(__name__)

# Defaults — override via configure_qwen() from main.py at startup.
QWEN_ENABLED = True
QWEN_URL = "http://127.0.0.1:1234/v1/chat/completions"
QWEN_MODEL = "qwen3-4b-instruct-2507"
QWEN_TIMEOUT = 10.0

DEFAULT_DAMAGE_MULT = 1.0
DEFAULT_HP_LOSS_MULT = 0.5

UNKNOWN_ENEMY_MECHANICS = (
    "scaling enemy - unknown mechanics - prefer aggression as default"
)

# Fallback when API hides piles until after the first draw (floor 1–3).
STARTER_DECK_IDS: dict[str, list[str]] = {
    "IRONCLAD": ["STRIKE"] * 5 + ["DEFEND"] * 4 + ["BASH"],
    "SILENT": ["STRIKE"] * 5 + ["DEFEND"] * 5 + ["NEUTRALIZE"],
    "DEFECT": ["STRIKE"] * 4 + ["DEFEND"] * 4 + ["ZAP", "DUALCAST"],
    "NECROBINDER": ["STRIKE"] * 5 + ["DEFEND"] * 4 + ["BONE_SHARDS"],
    "REGENT": ["STRIKE"] * 5 + ["DEFEND"] * 4 + ["VANGUARD"],
}

SYSTEM_PROMPT = (
    "You are a Slay the Spire 2 combat advisor. Use only the information provided. "
    "Do not rely on your own knowledge of the game. Respond with a JSON object only."
)

_settings: dict[str, Any] = {
    "enabled": QWEN_ENABLED,
    "combat_enabled": False,
    "macro_enabled": False,
    "macro_context_enabled": True,
    "url": QWEN_URL,
    "model": QWEN_MODEL,
    "timeout": QWEN_TIMEOUT,
    "log_full_prompt": False,
}


def configure_qwen(
    *,
    enabled: bool | None = None,
    combat_enabled: bool | None = None,
    macro_enabled: bool | None = None,
    macro_context_enabled: bool | None = None,
    url: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
    log_full_prompt: bool | None = None,
) -> None:
    """Apply runtime Qwen settings (called from main.py)."""
    if enabled is not None:
        _settings["enabled"] = bool(enabled)
    if combat_enabled is not None:
        _settings["combat_enabled"] = bool(combat_enabled)
    if macro_enabled is not None:
        _settings["macro_enabled"] = bool(macro_enabled)
    if macro_context_enabled is not None:
        _settings["macro_context_enabled"] = bool(macro_context_enabled)
    if url is not None:
        _settings["url"] = url
    if model is not None:
        _settings["model"] = model
    if timeout is not None:
        _settings["timeout"] = float(timeout)
    if log_full_prompt is not None:
        _settings["log_full_prompt"] = bool(log_full_prompt)


def is_qwen_combat_enabled() -> bool:
    return bool(_settings["enabled"]) and bool(_settings["combat_enabled"])


def is_qwen_macro_enabled() -> bool:
    return bool(_settings["enabled"]) and bool(_settings["macro_enabled"])


def is_qwen_macro_context_enabled() -> bool:
    return bool(_settings["enabled"]) and bool(_settings["macro_context_enabled"])


def qwen_settings() -> dict[str, Any]:
    return dict(_settings)


@dataclass
class FightMultipliers:
    strategy: str = "balanced"
    damage_mult: float = DEFAULT_DAMAGE_MULT
    hp_loss_mult: float = DEFAULT_HP_LOSS_MULT
    reasoning: str = ""
    source: str = "default"


@dataclass
class QwenAdvisor:
    """One synchronous Qwen call per fight; multipliers ready before the first decision."""

    _expert_enemies: dict[str, dict[str, Any]] = field(default_factory=dict)
    _multipliers: FightMultipliers = field(default_factory=FightMultipliers)
    _strategy_record: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._expert_enemies = _load_expert_enemies()
        _log_expert_enemy_load_status(self._expert_enemies)

    def begin_fight(
        self,
        state: dict[str, Any],
        *,
        combat_type: str,
        enemy_names: list[str],
        deck_card_ids: list[str] | None = None,
    ) -> FightMultipliers:
        """Block until Qwen responds (or timeout); call on first player turn."""
        self._multipliers = FightMultipliers()
        self._strategy_record = None

        if not is_qwen_combat_enabled():
            return self._multipliers

        try:
            result = self._fetch_strategy(
                state,
                combat_type=combat_type,
                enemy_names=list(enemy_names),
                deck_card_ids=deck_card_ids,
            )
        except Exception as exc:
            logger.debug("Qwen advisor fetch failed: %s", exc)
            result = FightMultipliers(source="default")

        self._apply_result(result)
        return self.get_multipliers()

    def get_multipliers(self) -> FightMultipliers:
        return FightMultipliers(
            strategy=self._multipliers.strategy,
            damage_mult=self._multipliers.damage_mult,
            hp_loss_mult=self._multipliers.hp_loss_mult,
            reasoning=self._multipliers.reasoning,
            source=self._multipliers.source,
        )

    def end_fight(self) -> dict[str, Any] | None:
        """Return strategy record for combat_summary."""
        return dict(self._strategy_record) if self._strategy_record else None

    def _apply_result(self, result: FightMultipliers) -> None:
        self._multipliers = result
        self._strategy_record = {
            "strategy": result.strategy,
            "damage_multiplier": result.damage_mult,
            "hp_loss_multiplier": result.hp_loss_mult,
            "reasoning": result.reasoning,
            "source": result.source,
        }
        if result.source == "qwen":
            logger.info(
                "Qwen strategy: %s (dmg×%.2f, hp×%.2f) - %s",
                result.strategy,
                result.damage_mult,
                result.hp_loss_mult,
                result.reasoning,
            )
        elif result.source == "timeout":
            logger.info("Qwen timeout - using default multipliers")

    def _fetch_strategy(
        self,
        state: dict[str, Any],
        *,
        combat_type: str,
        enemy_names: list[str],
        deck_card_ids: list[str] | None = None,
    ) -> FightMultipliers:
        user_prompt = build_combat_prompt(
            state,
            combat_type=combat_type,
            enemy_names=enemy_names,
            expert_enemies=self._expert_enemies,
            deck_card_ids=deck_card_ids,
        )
        try:
            raw = _call_qwen_api(user_prompt)
        except requests.Timeout:
            return FightMultipliers(source="timeout")
        except requests.RequestException:
            return FightMultipliers(source="default")
        except Exception as exc:
            logger.debug("Qwen API error: %s", exc)
            return FightMultipliers(source="default")

        parsed = parse_strategy_response(raw)
        if parsed is None:
            logger.debug("Qwen returned invalid JSON - using default multipliers")
            return FightMultipliers(source="default")
        return parsed

    def reload_expert_knowledge(self) -> None:
        self._expert_enemies = _load_expert_enemies()
        _log_expert_enemy_load_status(self._expert_enemies)


_advisor: QwenAdvisor | None = None


def get_qwen_advisor() -> QwenAdvisor:
    global _advisor
    if _advisor is None:
        _advisor = QwenAdvisor()
    return _advisor


def _enemy_name_lookup_keys(name: str) -> list[str]:
    """Normalized keys for matching API names like 'Twig Slime (S)' to knowledge entries."""
    keys: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        key = _normalize_name(raw)
        if key and key not in seen:
            seen.add(key)
            keys.append(key)

    add(name)
    base = re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()
    if base and base != name:
        add(base)
    return keys


def _log_expert_enemy_load_status(expert_enemies: dict[str, dict[str, Any]]) -> None:
    path = Path(EXPERT_KNOWLEDGE_PATH)
    if not path.exists():
        logger.warning(
            "Qwen: %s not found — enemy prompts use Codex + missing fallback only",
            path,
        )
        return
    if not expert_enemies:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            top_keys = list(raw.keys()) if isinstance(raw, dict) else []
        except (OSError, json.JSONDecodeError):
            top_keys = []
        logger.warning(
            "Qwen: expert_knowledge.json has 0 monster/enemy entries (top-level keys: %s). "
            "Enemy-specific strategy text is NOT injected until monsters are added.",
            top_keys,
        )
        return
    sample = sorted(expert_enemies.keys())[:8]
    logger.info(
        "Qwen: loaded %d enemy entries from expert_knowledge.json (e.g. %s)",
        len(expert_enemies),
        ", ".join(sample),
    )
    try:
        from sts2_agent.enemy_compendium import get_compendium_kb

        comp_n = len(get_compendium_kb().by_name)
        if comp_n:
            logger.info(
                "Qwen: learned compendium available (%d enemies in data/enemy_compendium.json)",
                comp_n,
            )
    except Exception:
        pass


def _load_expert_enemies() -> dict[str, dict[str, Any]]:
    """Load enemy entries from cache/expert_knowledge.json (normalized name keys)."""
    path = Path(EXPERT_KNOWLEDGE_PATH)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load expert knowledge for Qwen: %s", exc)
        return {}

    merged: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key in ("monsters", "enemies", "encounters"):
            block = raw.get(key)
            if isinstance(block, dict):
                for name, entry in block.items():
                    if isinstance(entry, dict):
                        for lookup_key in _enemy_name_lookup_keys(str(name)):
                            merged[lookup_key] = entry
        # Flat name -> entry at top level (excluding known meta keys)
        meta_keys = {
            "cards",
            "relics",
            "potions",
            "archetypes",
            "monsters",
            "enemies",
            "encounters",
            "archetypes_by_character",
            "fetched_at",
            "source",
            "counts",
        }
        for name, entry in raw.items():
            if name in meta_keys or not isinstance(entry, dict):
                continue
            if "tier" in entry or "notes" in entry or "patterns" in entry or "mechanics" in entry:
                for lookup_key in _enemy_name_lookup_keys(str(name)):
                    merged[lookup_key] = entry
    return merged


def character_key_from_state(state: dict[str, Any]) -> str:
    """IRONCLAD / SILENT / … from run or player fields."""
    run = state.get("run") or {}
    player = state.get("player") or {}
    for raw in (
        run.get("character"),
        run.get("character_id"),
        run.get("class"),
        player.get("character"),
        player.get("character_id"),
    ):
        if not raw:
            continue
        text = str(raw).strip().upper()
        if text.startswith("CHARACTER."):
            text = text.split(".", 1)[-1]
        if text.startswith("THE "):
            text = text[4:].strip()
        compact = text.replace(" ", "_")
        if compact in STARTER_DECK_IDS:
            return compact
        for key in STARTER_DECK_IDS:
            if key in compact:
                return key
    return ""


def starter_deck_ids_for_state(state: dict[str, Any]) -> list[str]:
    """Known starter lists for early floors when piles are not exposed yet."""
    run = state.get("run") or {}
    floor = int(run.get("floor") or 0)
    if floor > 3:
        return []
    key = character_key_from_state(state)
    if not key:
        return []
    return list(STARTER_DECK_IDS.get(key, []))


def lookup_learned_compendium(enemy_name: str) -> dict[str, Any] | None:
    """Agent-run enemy data from data/enemy_compendium.json."""
    try:
        from sts2_agent.enemy_compendium import get_compendium_kb

        return get_compendium_kb().lookup(enemy_name)
    except Exception as exc:
        logger.debug("Compendium lookup failed for %s: %s", enemy_name, exc)
        return None


def compendium_entry_for_prompt(raw: dict[str, Any]) -> dict[str, Any]:
    """Turn learned compendium blob into notes for the Qwen user prompt."""
    from sts2_agent.enemy_compendium import move_display_name

    parts: list[str] = []
    fights = int(raw.get("fight_count") or 0)
    if fights:
        parts.append(f"observed in {fights} agent fight(s)")

    cycle = raw.get("learned_cycle") or []
    if cycle:
        labels = [move_display_name(str(k)) for k in cycle[:6]]
        parts.append("intent cycle: " + " → ".join(labels))

    moves = raw.get("moves") or {}
    if isinstance(moves, dict) and moves:
        ranked = sorted(
            moves.items(),
            key=lambda kv: int((kv[1] or {}).get("seen_count") or 0),
            reverse=True,
        )[:5]
        move_bits: list[str] = []
        for move_key, meta in ranked:
            if not isinstance(meta, dict):
                continue
            label = move_display_name(str(move_key))
            dmg = int(meta.get("damage") or 0)
            tags = meta.get("tags") or []
            tag_str = ",".join(str(t) for t in tags[:3]) if tags else ""
            move_bits.append(f"{label} (dmg {dmg}{', ' + tag_str if tag_str else ''})")
        if move_bits:
            parts.append("moves seen: " + "; ".join(move_bits))

    category = str(raw.get("category") or "").strip()
    if category and category.lower() != "unknown":
        parts.append(f"category: {category}")

    role = raw.get("role")
    if role:
        parts.append(f"role: {role}")

    return {"notes": ". ".join(parts) if parts else "learned from agent runs (sparse)"}


def lookup_enemy_knowledge(
    enemy_name: str,
    expert_enemies: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Return (entry, source): expert_knowledge → learned_compendium → codex → missing."""
    for key in _enemy_name_lookup_keys(enemy_name):
        if key in expert_enemies:
            return expert_enemies[key], "expert_knowledge"

    comp = lookup_learned_compendium(enemy_name)
    if comp and (comp.get("moves") or comp.get("learned_cycle") or int(comp.get("fight_count") or 0)):
        return compendium_entry_for_prompt(comp), "learned_compendium"

    try:
        kb = get_knowledge()
        for key in _enemy_name_lookup_keys(enemy_name):
            monster = kb.monsters_by_name.get(key)
            if monster:
                desc = str(monster.get("description") or monster.get("desc") or "").strip()
                if desc:
                    return {
                        "name": monster.get("name") or enemy_name,
                        "description": desc,
                    }, "codex"
    except Exception:
        pass
    return None, "missing"


def _format_enemy_section(name: str, entry: dict[str, Any] | None, source: str) -> str:
    tag = f"[{source}]"
    if not entry:
        return f"- {name}: {tag} {UNKNOWN_ENEMY_MECHANICS}"
    notes = entry.get("notes") or entry.get("mechanics") or entry.get("patterns") or ""
    desc = entry.get("description") or entry.get("desc") or ""
    tier = entry.get("tier")
    lines = [f"- {name}: {tag}"]
    if tier:
        lines.append(f"  tier: {tier}")
    if notes:
        lines.append(f"  mechanics: {notes}")
    elif desc:
        lines.append(f"  description: {desc}")
    elif source == "learned_compendium":
        lines.append(f"  mechanics: {json.dumps(entry, ensure_ascii=False)[:400]}")
    else:
        lines.append(f"  data: {json.dumps(entry, ensure_ascii=False)[:400]}")
    return "\n".join(lines)


def build_combat_prompt(
    state: dict[str, Any],
    *,
    combat_type: str,
    enemy_names: list[str],
    expert_enemies: dict[str, dict[str, Any]],
    deck_card_ids: list[str] | None = None,
) -> str:
    run = state.get("run") or {}
    player = state.get("player") or {}
    floor = int(run.get("floor") or 0)
    act = int(run.get("act") or 1)
    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    energy = player.get("energy", player.get("mana", "?"))

    enemy_sections: list[str] = []
    found_expert: list[str] = []
    found_compendium: list[str] = []
    found_codex: list[str] = []
    missing: list[str] = []

    for name in enemy_names:
        entry, source = lookup_enemy_knowledge(name, expert_enemies)
        if source == "expert_knowledge":
            found_expert.append(name)
        elif source == "learned_compendium":
            found_compendium.append(name)
        elif source == "codex":
            found_codex.append(name)
        else:
            missing.append(name)

        enemy_sections.append(_format_enemy_section(name, entry, source))

    logger.info(
        "Qwen knowledge injection: expert=%s compendium=%s codex=%s missing=%s",
        found_expert or "(none)",
        found_compendium or "(none)",
        found_codex or "(none)",
        missing or "(none)",
    )

    deck_lines = _deck_lines(state, deck_card_ids=deck_card_ids)
    starter_ids = starter_deck_ids_for_state(state) if not deck_lines and not deck_card_ids else []
    if deck_lines:
        deck_source = "live piles"
    elif deck_card_ids:
        deck_source = "cached card ids"
    elif starter_ids:
        deck_source = f"starter deck ({character_key_from_state(state) or 'unknown'})"
        if not deck_lines:
            deck_lines = [f"- {cid}" for cid in starter_ids]
    else:
        deck_source = "unavailable"
    logger.info(
        "Qwen deck context: %d cards (%s)",
        len(deck_lines),
        deck_source,
    )

    return (
        f"Combat type: {combat_type}\n"
        f"Floor: {floor}, Act: {act}\n"
        f"Player HP: {hp}/{max_hp}\n"
        f"Energy: {energy}\n\n"
        "Enemies:\n"
        + ("\n".join(enemy_sections) if enemy_sections else "- (unknown)\n")
        + "\n\nCards will be drawn at turn start. Deck composition is:\n"
        + ("\n".join(deck_lines) if deck_lines else "- (empty)\n")
        + "\n\nBased on this combat situation, return a JSON object with:\n"
        "{\n"
        '  "strategy": "aggressive" | "balanced" | "defensive",\n'
        '  "damage_multiplier": float between 0.5 and 2.0,\n'
        '  "hp_loss_multiplier": float between 0.25 and 1.0,\n'
        '  "reasoning": "one sentence explanation"\n'
        "}"
    )


def _deck_lines(
    state: dict[str, Any],
    *,
    deck_card_ids: list[str] | None = None,
) -> list[str]:
    """All deck cards from every pile; fallback to cached ids from the run."""
    from sts2_agent.scorer import card_name, deck_cards

    seen: set[str] = set()
    lines: list[str] = []
    for card in deck_cards(state):
        label = card_name(card).strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {label}")

    if lines:
        return lines

    if deck_card_ids:
        try:
            kb = get_knowledge()
        except Exception:
            kb = None
        for cid in deck_card_ids:
            label = str(cid).strip()
            if kb:
                codex = kb.lookup_card(label)
                if codex and codex.get("name"):
                    label = str(codex["name"])
            key = label.lower()
            if key and key not in seen:
                seen.add(key)
                lines.append(f"- {label}")
    return lines


def _log_full_qwen_prompt(user_prompt: str, system_prompt: str) -> None:
    if not _settings.get("log_full_prompt"):
        return
    logger.debug(
        "Qwen full prompt (%s)\n======== SYSTEM ========\n%s\n======== USER ========\n%s\n======== END ========",
        _settings.get("model"),
        system_prompt,
        user_prompt,
    )


def _call_qwen_api(user_prompt: str, *, system_prompt: str | None = None) -> str:
    system = system_prompt or SYSTEM_PROMPT
    _log_full_qwen_prompt(user_prompt, system)
    payload = {
        "model": _settings["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    response = requests.post(
        _settings["url"],
        json=payload,
        timeout=_settings["timeout"],
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("empty choices")
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("empty content")
    return content.strip()


def parse_strategy_response(text: str) -> FightMultipliers | None:
    """Parse model JSON (possibly wrapped in markdown fences)."""
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.I)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None

    strategy = str(obj.get("strategy") or "balanced").lower().strip()
    if strategy not in ("aggressive", "balanced", "defensive"):
        strategy = "balanced"

    try:
        damage_mult = float(obj.get("damage_multiplier", DEFAULT_DAMAGE_MULT))
    except (TypeError, ValueError):
        damage_mult = DEFAULT_DAMAGE_MULT
    try:
        hp_loss_mult = float(obj.get("hp_loss_multiplier", DEFAULT_HP_LOSS_MULT))
    except (TypeError, ValueError):
        hp_loss_mult = DEFAULT_HP_LOSS_MULT

    damage_mult = max(0.5, min(2.0, damage_mult))
    hp_loss_mult = max(0.25, min(1.0, hp_loss_mult))
    reasoning = str(obj.get("reasoning") or "").strip()

    return FightMultipliers(
        strategy=strategy,
        damage_mult=damage_mult,
        hp_loss_mult=hp_loss_mult,
        reasoning=reasoning,
        source="qwen",
    )
