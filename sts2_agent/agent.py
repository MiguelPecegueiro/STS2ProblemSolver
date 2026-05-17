"""Route game state to handlers; optional BC/PPO policy with rules fallback."""

from __future__ import annotations

import json
import logging

from sts2_agent import card_select, combat, event, map as map_handler, rewards, rest, shop
from sts2_agent.action_validate import normalize_policy_action, validate_policy_action
from sts2_agent.data_pipeline import observe_state
from sts2_agent.card_select import effective_selected_indices
from sts2_agent.state_parse import (
    event_in_dialogue,
    extract_event_options,
    extract_rest_options,
    extract_shop_items,
    extract_treasure_relics,
    get_card_select_screen,
    treasure_can_proceed,
    get_event_screen,
    get_shop_screen,
    is_card_select_active,
)
from sts2_agent.agent_types import Decision

logger = logging.getLogger(__name__)

COMBAT_TYPES = combat.COMBAT_STATE_TYPES | {"hand_select"}

_policy_enabled = False
_policy_combat_only = False


def configure_policy(*, enabled: bool = False, combat_only: bool = False) -> None:
    global _policy_enabled, _policy_combat_only
    _policy_enabled = enabled
    _policy_combat_only = combat_only and not enabled


def policy_active_for_state(state_type: str) -> bool:
    if not _policy_enabled and not _policy_combat_only:
        return False
    if _policy_enabled:
        return True
    return state_type in COMBAT_TYPES


def _decide_rules(state: dict) -> Decision:
    state_type = str(state.get("state_type") or "").lower()

    if state_type == "hand_select":
        action, reasons = combat.decide_combat(state)
        return Decision(action, reasons)

    if is_card_select_active(state):
        action, reasons = card_select.decide_card_select(state)
        return Decision(action, reasons)

    if state_type in combat.COMBAT_STATE_TYPES:
        action, reasons = combat.decide_combat(state)
        return Decision(action, reasons)

    if state_type == "map":
        action, reasons = map_handler.decide_map(state)
        return Decision(action, reasons)

    if state_type == "card_reward":
        action, reasons = rewards.decide_card_reward(state)
        return Decision(action, reasons)

    if state_type == "rewards":
        action, reasons = rewards.decide_rewards(state)
        return Decision(action, reasons)

    if state_type == "treasure":
        action, reasons = rewards.decide_treasure(state)
        return Decision(action, reasons)

    if state_type == "rest_site":
        action, reasons = rest.decide_rest_site(state)
        return Decision(action, reasons)

    if state_type in ("shop", "fake_merchant"):
        action, reasons = shop.decide_shop(state)
        return Decision(action, reasons)

    if state_type == "event" or get_event_screen(state):
        action, reasons = event.decide_event(state)
        return Decision(action, reasons)

    if state_type in ("menu", "game_over"):
        return Decision(None, ["menu flow handled in main loop"])

    logger.debug("No handler for state_type=%s", state_type)
    return Decision(None, [f"unhandled state_type: {state_type}"])


def _decide_policy(state: dict, state_type: str) -> Decision | None:
    try:
        from training.inference import get_policy

        policy = get_policy()
        action, policy_reasons = policy.predict(state)
    except Exception as exc:
        logger.warning("Policy load/predict failed: %s", exc)
        return None

    prefix = ["policy_net"]
    if action is None:
        return Decision(None, prefix + policy_reasons)

    action = normalize_policy_action(state, action)

    try:
        valid, reason = validate_policy_action(state, action)
    except Exception as exc:
        logger.warning("Policy validation error: %s - %s", exc, action)
        return Decision(
            None,
            prefix + [f"validation error: {exc}"] + policy_reasons,
        )

    if valid:
        return Decision(action, prefix + policy_reasons)

    logger.debug("Policy action rejected (%s): %s", reason, action)
    return Decision(
        None,
        prefix + [f"invalid: {reason}"] + policy_reasons,
    )


def decide(state: dict) -> Decision:
    """Return action plus reasoning for the current game state."""
    observe_state(state)

    state_type = str(state.get("state_type") or "").lower()
    if not state_type:
        logger.warning("Missing state_type in game state")
        return Decision(None, ["missing state_type"])

    # Policy does not handle multi-step overlays (hand_select / card_select).
    if is_card_select_active(state) or state_type == "hand_select":
        return _decide_rules(state)

    if policy_active_for_state(state_type):
        pot_reasons: list[str] = []
        if state_type in combat.COMBAT_STATE_TYPES:
            pot_action, pot_reasons = combat.decide_combat_potion(state)
            if pot_action:
                return Decision(pot_action, ["potion priority (before policy)"] + pot_reasons)

        policy_decision = _decide_policy(state, state_type)
        if policy_decision is not None and policy_decision.action is not None:
            merged = pot_reasons + policy_decision.reasons
            return Decision(policy_decision.action, merged)

        fallback_reason = ["rules fallback"]
        if policy_decision is not None and policy_decision.reasons:
            fallback_reason.extend(policy_decision.reasons)

        rule_decision = _decide_rules(state)
        return Decision(
            rule_decision.action,
            fallback_reason + rule_decision.reasons,
        )

    return _decide_rules(state)


def decide_action(state: dict) -> dict | None:
    return decide(state).action


def state_fingerprint(state: dict) -> str:
    from sts2_agent.state_parse import extract_card_reward_cards, extract_map_choices, extract_reward_items

    payload = {
        "state_type": state.get("state_type"),
        "run": state.get("run"),
        "battle": {
            "turn": (state.get("battle") or {}).get("turn"),
            "round": (state.get("battle") or {}).get("round"),
            "is_play_phase": (state.get("battle") or {}).get("is_play_phase"),
        },
        "player": {
            "hp": (state.get("player") or {}).get("hp"),
            "energy": (state.get("player") or {}).get("energy"),
            "hand": [
                (c.get("name") if isinstance(c, dict) else c)
                for c in ((state.get("player") or {}).get("hand") or [])
            ],
        },
        "enemies": [
            {
                "id": e.get("entity_id"),
                "hp": e.get("hp"),
                "intents": e.get("intents"),
            }
            for e in ((state.get("battle") or {}).get("enemies") or [])
        ],
        "map_choices": [
            {
                "index": o.get("index"),
                "type": o.get("type"),
            }
            for o in extract_map_choices(state)
        ],
        "reward_items": [
            {
                "index": r.get("index"),
                "type": r.get("type"),
                "claimed": r.get("claimed"),
            }
            for r in extract_reward_items(state)
        ],
        "card_reward_cards": [
            c.get("name") if isinstance(c, dict) else c
            for c in extract_card_reward_cards(state)
        ],
        "hand_select": {
            "mode": (state.get("hand_select") or {}).get("mode"),
            "can_confirm": (state.get("hand_select") or {}).get("can_confirm"),
            "selected": [
                c.get("name") for c in ((state.get("hand_select") or {}).get("selected_cards") or [])
            ],
        },
        "card_select": {
            "screen_type": (get_card_select_screen(state) or {}).get("screen_type"),
            "can_confirm": (get_card_select_screen(state) or {}).get("can_confirm"),
            "preview": (get_card_select_screen(state) or {}).get("preview_showing"),
            "selected": sorted(
                effective_selected_indices(get_card_select_screen(state) or {})
            ),
        },
        "rest_site": {
            "can_proceed": (state.get("rest_site") or {}).get("can_proceed"),
            "options": [
                {
                    "index": o.get("index"),
                    "id": o.get("id"),
                    "enabled": o.get("is_enabled"),
                }
                for o in extract_rest_options(state)
            ],
        },
        "shop": {
            "can_proceed": (get_shop_screen(state) or {}).get("can_proceed"),
            "gold": (state.get("player") or {}).get("gold"),
            "items": [
                {
                    "index": i.get("index"),
                    "category": i.get("category"),
                    "stocked": i.get("is_stocked"),
                    "name": i.get("card_name") or i.get("relic_name") or i.get("potion_name"),
                }
                for i in (get_shop_screen(state) or {}).get("items") or []
                if isinstance(i, dict)
            ],
            "purchasable": len(extract_shop_items(state)),
        },
        "treasure": {
            "can_proceed": treasure_can_proceed(state),
            "relics": [
                r.get("name") for r in extract_treasure_relics(state)[:6]
            ],
        },
        "event": {
            "in_dialogue": event_in_dialogue(state),
            "body": str((get_event_screen(state) or {}).get("body") or "")[:200],
            "options": [
                {
                    "index": o.get("index"),
                    "title": o.get("title"),
                    "is_proceed": o.get("is_proceed"),
                    "was_chosen": o.get("was_chosen"),
                    "locked": o.get("is_locked"),
                }
                for o in extract_event_options(state)
            ],
        },
    }
    return json.dumps(payload, sort_keys=True, default=str)
