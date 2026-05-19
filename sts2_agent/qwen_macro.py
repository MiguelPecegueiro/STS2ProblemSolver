"""Qwen advisor for non-combat screens: map, rewards, rest, shop, events."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from sts2_agent.action_validate import validate_policy_action
from sts2_agent.agent_types import Decision
from sts2_agent.knowledge import get_knowledge
from sts2_agent.qwen_advisor import _call_qwen_api, is_qwen_macro_enabled
from sts2_agent.scorer import card_name, deck_cards
from sts2_agent.state_parse import (
    card_reward_can_skip,
    card_reward_index,
    event_in_dialogue,
    event_option_index,
    event_option_label,
    extract_card_reward_cards,
    extract_event_options,
    extract_map_choices,
    extract_rest_options,
    extract_reward_items,
    extract_shop_items,
    get_event_screen,
    map_choice_index,
    map_choice_room_type,
    rest_option_index,
    reward_item_index,
    shop_item_index,
)

logger = logging.getLogger(__name__)

MACRO_SYSTEM_PROMPT = (
    "You are a Slay the Spire 2 run advisor for non-combat decisions. "
    "Use only the information provided. Respond with a single JSON object only."
)

MACRO_STATE_TYPES = frozenset(
    {
        "map",
        "card_reward",
        "rewards",
        "treasure",
        "rest_site",
        "shop",
        "fake_merchant",
        "event",
    }
)

_last_trace: dict[str, Any] | None = None


def macro_state_types() -> frozenset[str]:
    return MACRO_STATE_TYPES


def clear_macro_qwen_trace() -> None:
    global _last_trace
    _last_trace = None


def pop_macro_qwen_trace() -> dict[str, Any] | None:
    """Return and clear the trace from the latest macro Qwen API attempt."""
    global _last_trace
    trace = _last_trace
    _last_trace = None
    return dict(trace) if trace else None


def _begin_macro_trace(state_type: str, user_prompt: str) -> None:
    global _last_trace
    _last_trace = {
        "state_type": state_type,
        "system_prompt": MACRO_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }


def _update_macro_trace(**fields: Any) -> None:
    if _last_trace is not None:
        _last_trace.update(fields)


def _parse_json_object(text: str) -> dict[str, Any] | None:
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
    return obj if isinstance(obj, dict) else None


def _player_context(state: dict) -> str:
    run = state.get("run") or {}
    player = state.get("player") or {}
    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    gold = int(player.get("gold") or 0)
    floor = int(run.get("floor") or 0)
    act = int(run.get("act") or 1)
    relics = [
        str(r.get("name") or r.get("id") or r)
        for r in (player.get("relics") or [])
        if r
    ]
    return (
        f"Floor {floor}, Act {act}\n"
        f"HP: {hp}/{max_hp} ({hp / max_hp:.0%})\n"
        f"Gold: {gold}\n"
        f"Relics: {', '.join(relics) if relics else '(none)'}\n"
    )


def _deck_context(state: dict) -> str:
    lines: list[str] = []
    for card in deck_cards(state):
        label = card_name(card)
        if not label:
            continue
        kb = get_knowledge()
        tier = kb.expert_card_tier(label) or "?"
        notes = ""
        entry = kb._expert_entry(label, kb.expert_cards)  # noqa: SLF001
        if entry and entry.get("notes"):
            notes = f" — {str(entry['notes'])[:120]}"
        lines.append(f"- [{tier}] {label}{notes}")
    if not lines:
        return "Deck: (unknown)\n"
    return "Deck:\n" + "\n".join(lines[:40]) + "\n"


def _card_reward_prompt(state: dict) -> str:
    kb = get_knowledge()
    cards = extract_card_reward_cards(state)
    lines: list[str] = []
    for i, card in enumerate(cards):
        idx = card_reward_index(card, i)
        name = card_name(card)
        tier = kb.expert_card_tier(name) or "?"
        lines.append(f"  [{idx}] {name} (tier {tier})")
    skip = "yes" if card_reward_can_skip(state) else "no"
    return (
        f"{_player_context(state)}\n{_deck_context(state)}\n"
        "Card reward — pick one card or skip.\n"
        + ("\n".join(lines) if lines else "  (no cards)\n")
        + f"\nCan skip: {skip}\n\n"
        'Return JSON: {"action":"select_card_reward","card_index":N} or '
        '{"action":"skip_card_reward","reasoning":"..."}\n'
    )


def _map_prompt(state: dict) -> str:
    choices = extract_map_choices(state)
    lines: list[str] = []
    for i, opt in enumerate(choices):
        idx = map_choice_index(opt, i)
        room = map_choice_room_type(opt)
        lines.append(f"  [{idx}] {room}")
    return (
        f"{_player_context(state)}\n{_deck_context(state)}\n"
        "Map — choose the next node index.\n"
        + ("\n".join(lines) if lines else "  (no paths)\n")
        + '\n\nReturn JSON: {"action":"choose_map_node","index":N,"reasoning":"..."}\n'
    )


def _rest_prompt(state: dict) -> str:
    options = extract_rest_options(state)
    lines: list[str] = []
    for i, opt in enumerate(options):
        idx = rest_option_index(opt, i)
        label = str(opt.get("id") or opt.get("label") or opt.get("name") or f"option_{idx}")
        enabled = opt.get("is_enabled", True)
        lines.append(f"  [{idx}] {label} (enabled={enabled})")
    return (
        f"{_player_context(state)}\n{_deck_context(state)}\n"
        "Rest site — choose one option index.\n"
        + ("\n".join(lines) if lines else "  (no options)\n")
        + '\n\nReturn JSON: {"action":"choose_rest_option","index":N,"reasoning":"..."}\n'
        'Or {"action":"proceed"} if rest is complete.\n'
    )


def _shop_prompt(state: dict) -> str:
    items = extract_shop_items(state)
    lines: list[str] = []
    for i, item in enumerate(items):
        idx = shop_item_index(item, i)
        name = (
            item.get("card_name")
            or item.get("relic_name")
            or item.get("potion_name")
            or item.get("name")
            or "?"
        )
        price = item.get("price") or item.get("cost") or "?"
        stocked = item.get("is_stocked", True)
        lines.append(f"  [{idx}] {name} — {price}g (stocked={stocked})")
    return (
        f"{_player_context(state)}\n{_deck_context(state)}\n"
        "Shop — buy one item by index, or leave.\n"
        + ("\n".join(lines) if lines else "  (no items)\n")
        + '\n\nReturn JSON: {"action":"shop_purchase","index":N} or '
        '{"action":"proceed","reasoning":"..."}\n'
    )


def _event_prompt(state: dict) -> str:
    screen = get_event_screen(state) or {}
    body = str(screen.get("body") or screen.get("text") or "")[:800]
    event_name = str(screen.get("event_name") or screen.get("event_id") or "event")
    if event_in_dialogue(state):
        return (
            f"Event: {event_name}\n{body}\n\n"
            'Return JSON: {"action":"advance_dialogue","reasoning":"..."}\n'
        )
    options = extract_event_options(state)
    lines: list[str] = []
    for i, opt in enumerate(options):
        idx = event_option_index(opt, i)
        title = event_option_label(opt) or f"option_{idx}"
        locked = opt.get("is_locked", False)
        lines.append(f"  [{idx}] {title} (locked={locked})")
    return (
        f"{_player_context(state)}\n"
        f"Event: {event_name}\n{body}\n\n"
        "Choose an option index or advance dialogue.\n"
        + ("\n".join(lines) if lines else "  (no options)\n")
        + '\n\nReturn JSON: {"action":"choose_event_option","index":N} or '
        '{"action":"advance_dialogue","reasoning":"..."}\n'
    )


def _rewards_prompt(state: dict) -> str:
    items = extract_reward_items(state)
    lines: list[str] = []
    for i, item in enumerate(items):
        idx = reward_item_index(item, i)
        itype = str(item.get("type") or "?")
        claimed = item.get("claimed", False)
        lines.append(f"  [{idx}] {itype} claimed={claimed}")
    return (
        f"{_player_context(state)}\n"
        "Rewards screen — claim next reward by index or proceed.\n"
        + ("\n".join(lines) if lines else "  (no items)\n")
        + '\n\nReturn JSON: {"action":"claim_reward","index":N} or '
        '{"action":"proceed"} or {"action":"discard_potion","slot":N}\n'
    )


def _treasure_prompt(state: dict) -> str:
    from sts2_agent.state_parse import extract_treasure_relics

    relics = extract_treasure_relics(state)
    lines = [f"  [{i}] {r.get('name', '?')}" for i, r in enumerate(relics) if isinstance(r, dict)]
    return (
        f"{_player_context(state)}\n"
        "Treasure — take relic by index or proceed.\n"
        + ("\n".join(lines) if lines else "  (no relics)\n")
        + '\n\nReturn JSON: {"action":"claim_treasure_relic","index":N} or '
        '{"action":"proceed","reasoning":"..."}\n'
    )


def build_macro_prompt(state: dict, state_type: str) -> str | None:
    st = state_type.lower()
    if st == "map":
        return _map_prompt(state)
    if st == "card_reward":
        return _card_reward_prompt(state)
    if st == "rewards":
        return _rewards_prompt(state)
    if st == "treasure":
        return _treasure_prompt(state)
    if st == "rest_site":
        return _rest_prompt(state)
    if st in ("shop", "fake_merchant"):
        return _shop_prompt(state)
    if st == "event":
        return _event_prompt(state)
    return None


def _action_from_response(obj: dict[str, Any]) -> dict[str, Any] | None:
    name = str(obj.get("action") or "").strip()
    if not name:
        return None
    action: dict[str, Any] = {"action": name}
    if name in ("select_card_reward", "choose_map_node", "choose_rest_option", "shop_purchase"):
        if obj.get("card_index") is not None:
            action["card_index"] = int(obj["card_index"])
        if obj.get("index") is not None:
            action["index"] = int(obj["index"])
    elif name == "choose_event_option":
        if obj.get("index") is not None:
            action["index"] = int(obj["index"])
    elif name == "claim_reward":
        if obj.get("index") is not None:
            action["index"] = int(obj["index"])
    elif name == "claim_treasure_relic":
        if obj.get("index") is not None:
            action["index"] = int(obj["index"])
    elif name == "discard_potion":
        if obj.get("slot") is not None:
            action["slot"] = int(obj["slot"])
    elif name in ("skip_card_reward", "proceed", "advance_dialogue"):
        pass
    else:
        return None
    return action


def fetch_macro_qwen_context(state: dict) -> list[str]:
    """Optional Qwen advisory text for logging; does not select actions."""
    from sts2_agent.qwen_advisor import is_qwen_macro_context_enabled
    from training.ppo_macro import ppo_macro_state_types

    if not is_qwen_macro_context_enabled():
        return []

    state_type = str(state.get("state_type") or "").lower()
    if state_type not in ppo_macro_state_types():
        return []

    prompt = build_macro_prompt(state, state_type)
    if not prompt:
        return []

    _begin_macro_trace(state_type, prompt)

    try:
        raw = _call_qwen_api(prompt, system_prompt=MACRO_SYSTEM_PROMPT)
    except requests.Timeout:
        _update_macro_trace(error="timeout", source="context_error")
        return ["qwen_context: timeout"]
    except requests.RequestException as exc:
        _update_macro_trace(error=str(exc), source="context_error")
        return [f"qwen_context: api error ({exc})"]
    except Exception as exc:
        _update_macro_trace(error=str(exc), source="context_error")
        return [f"qwen_context: error ({exc})"]

    _update_macro_trace(response=raw)
    obj = _parse_json_object(raw)
    if not obj:
        _update_macro_trace(parsed=None, source="context_invalid_json")
        return ["qwen_context: invalid JSON"]

    _update_macro_trace(parsed=obj, source="context")
    reasoning = str(obj.get("reasoning") or "").strip()
    suggested = _action_from_response(obj)
    lines = ["qwen_context: advisory only (PPO decides)"]
    if reasoning:
        lines.append(f"qwen_context reasoning: {reasoning}")
    if suggested:
        lines.append(f"qwen_context would pick: {suggested}")
    return lines


def try_qwen_macro_decide(state: dict) -> Decision | None:
    """Return a validated macro decision, or None to use rules fallback."""
    if not is_qwen_macro_enabled():
        return None

    state_type = str(state.get("state_type") or "").lower()
    if state_type not in MACRO_STATE_TYPES:
        return None

    prompt = build_macro_prompt(state, state_type)
    if not prompt:
        return None

    _begin_macro_trace(state_type, prompt)

    try:
        raw = _call_qwen_api(prompt, system_prompt=MACRO_SYSTEM_PROMPT)
    except requests.Timeout:
        logger.info("Qwen macro timeout on %s", state_type)
        _update_macro_trace(error="timeout", source="error")
        return None
    except requests.RequestException as exc:
        logger.debug("Qwen macro API error on %s: %s", state_type, exc)
        _update_macro_trace(error=str(exc), source="error")
        return None
    except Exception as exc:
        logger.debug("Qwen macro failed on %s: %s", state_type, exc)
        _update_macro_trace(error=str(exc), source="error")
        return None

    _update_macro_trace(response=raw)

    obj = _parse_json_object(raw)
    if not obj:
        logger.debug("Qwen macro invalid JSON on %s", state_type)
        _update_macro_trace(parsed=None, source="invalid_json")
        return None

    _update_macro_trace(parsed=obj)

    action = _action_from_response(obj)
    if not action:
        logger.debug("Qwen macro unmapped action on %s: %s", state_type, obj)
        _update_macro_trace(source="unmapped_action")
        return None

    valid, reason = validate_policy_action(state, action)
    if not valid:
        logger.info("Qwen macro rejected (%s): %s", reason, action)
        _update_macro_trace(action=action, validation_error=reason, source="rejected")
        return None

    reasoning = str(obj.get("reasoning") or "").strip()
    reasons = [f"qwen_macro: {state_type}", f"action={action}"]
    if reasoning:
        reasons.append(reasoning)
    _update_macro_trace(action=action, source="qwen")
    logger.info("Qwen macro %s → %s", state_type, action.get("action"))
    return Decision(action, reasons)
