"""Event dialogue and option selection."""

from __future__ import annotations

from sts2_agent.data_pipeline import record_handler_decision
from sts2_agent.state_parse import (
    event_in_dialogue,
    event_option_index,
    event_option_label,
    extract_event_options,
)

_RISKY_KEYWORDS = frozenset(
    {"lose", "hp", "damage", "curse", "injury", "pain", "gold", "pay", "bet"}
)
_SAFE_KEYWORDS = frozenset(
    {"gain", "heal", "card", "relic", "gold", "upgrade", "max_hp", "potion"}
)


def record_training(state: dict, action: dict | None, reasoning: list[str]) -> None:
    record_handler_decision(state, action, reasoning, handler="event")


def decide_event(state: dict) -> tuple[dict | None, list[str]]:
    """
    Event flow (STS2MCP):
    1. advance_dialogue while event.in_dialogue
    2. choose_event_option for real choices and Proceed (is_proceed) buttons
    """
    screen = state.get("event") or {}
    event_name = str(screen.get("event_name") or screen.get("event_id") or "event")
    reasons: list[str] = [f"event: {event_name}"]

    if event_in_dialogue(state):
        return {"action": "advance_dialogue"}, reasons + ["advance ancient/event dialogue"]

    options = extract_event_options(state)
    if not options:
        return {"action": "advance_dialogue"}, reasons + [
            "no options visible - try advance_dialogue"
        ]

    proceed_options = [
        o
        for o in options
        if o.get("is_proceed") or _is_continue_option(o)
    ]
    if proceed_options:
        choice = proceed_options[0]
        api_index = event_option_index(choice, 0)
        label = event_option_label(choice) or "Proceed"
        return {"action": "choose_event_option", "index": api_index}, reasons + [
            f"proceed/leave event - choose option {api_index} ({label})"
        ]

    choosable = [o for o in options if not o.get("was_chosen")]
    if not choosable:
        choice = options[0]
        api_index = event_option_index(choice, 0)
        return {"action": "choose_event_option", "index": api_index}, reasons + [
            f"only prior choices remain - pick option {api_index}"
        ]

    player = state.get("player") or {}
    hp_ratio = _hp_ratio(player)
    gold = int(player.get("gold") or 0)

    scored: list[tuple[int, float, str]] = []
    for list_idx, option in enumerate(choosable):
        api_index = event_option_index(option, list_idx)
        label = event_option_label(option)
        score = _score_event_option(option, hp_ratio, gold)
        scored.append((api_index, score, label))
        reasons.append(f"  [{api_index}] {label}: {score:.1f}")

    scored.sort(key=lambda x: x[1], reverse=True)
    best_index, best_score, best_label = scored[0]
    reasons.append(f"choose_event_option index {best_index} ({best_label}, score {best_score:.1f})")
    return {"action": "choose_event_option", "index": best_index}, reasons


def _score_event_option(option: dict | str, hp_ratio: float, gold: int) -> float:
    text = event_option_label(option).lower() if isinstance(option, dict) else str(option).lower()
    score = 40.0

    for word in _SAFE_KEYWORDS:
        if word in text:
            score += 12

    for word in _RISKY_KEYWORDS:
        if word in text:
            score -= 18

    if hp_ratio < 0.4 and any(w in text for w in ("hp", "damage", "lose")):
        score -= 40

    if gold < 50 and "pay" in text:
        score -= 30

    if "leave" in text or "ignore" in text:
        score -= 5

    if "?" in text or "fight" in text:
        score -= 10 if hp_ratio < 0.5 else 5

    return score


def _hp_ratio(player: dict) -> float:
    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    return hp / max_hp if max_hp else 1.0


def _is_continue_option(option: dict) -> bool:
    label = event_option_label(option).lower()
    return any(
        word in label
        for word in (
            "proceed",
            "continue",
            "leave",
            "exit",
            "done",
            "next",
            "ok",
            "okay",
        )
    )
