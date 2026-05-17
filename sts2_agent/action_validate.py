"""Validate policy-predicted actions before sending to the game API."""

from __future__ import annotations

from typing import Any

from sts2_agent import combat
from sts2_agent.card_select import effective_required_count, effective_selected_indices
from sts2_agent.state_parse import (
    card_reward_can_skip,
    card_reward_index,
    card_select_required_count,
    event_has_proceed_option,
    event_in_dialogue,
    event_option_index,
    extract_card_reward_cards,
    extract_event_options,
    extract_map_choices,
    extract_rest_options,
    extract_reward_items,
    extract_shop_items,
    extract_treasure_relics,
    rest_can_proceed,
    treasure_can_proceed,
    get_card_select_screen,
    get_shop_screen,
    is_card_select_active,
    living_enemies,
    map_choice_index,
    rest_option_index,
    reward_item_index,
    rewards_can_proceed,
)


def _hand_card_by_index(hand: list, card_index: int) -> dict | None:
    for list_idx, card in enumerate(hand):
        if not isinstance(card, dict):
            continue
        api_idx = int(card.get("index") if card.get("index") is not None else list_idx)
        if api_idx == card_index:
            return card
    if 0 <= card_index < len(hand) and isinstance(hand[card_index], dict):
        return hand[card_index]
    return None


def _enemy_entity_ids(state: dict) -> set[str]:
    battle = state.get("battle") or {}
    ids: set[str] = set()
    for enemy in living_enemies(battle):
        eid = enemy.get("entity_id") or enemy.get("id")
        if eid:
            ids.add(str(eid))
    return ids


def _card_needs_target(card: dict) -> bool:
    from sts2_agent.knowledge import get_knowledge

    kb = get_knowledge()
    from sts2_agent.combat import _needs_target

    return _needs_target(card, kb)


def _validate_play_card(state: dict, action: dict) -> tuple[bool, str]:
    state_type = str(state.get("state_type") or "").lower()
    if state_type not in combat.COMBAT_STATE_TYPES:
        return False, f"play_card invalid on state_type={state_type}"
    battle = state.get("battle") or {}
    if battle.get("is_play_phase") is False:
        return False, "not play phase"
    if str(battle.get("turn") or "").lower() not in ("", "player"):
        return False, "not player turn"

    player = state.get("player") or {}
    hand = player.get("hand") or []
    card_index = int(action.get("card_index", -1))
    card = _hand_card_by_index(hand, card_index)
    if card is None:
        return False, f"card_index {card_index} not in hand"

    if card.get("can_play") is False or card.get("playable") is False:
        return False, f"card {card_index} not playable"

    energy = int(player.get("energy") or player.get("current_energy") or 0)
    from sts2_agent.scorer import _card_cost

    cost = _card_cost(card)
    # X-cost cards report cost "X"; trust can_play when cost is unbounded.
    if cost < 99 and cost > energy:
        return False, f"card cost {cost} > energy {energy}"

    target = action.get("target")
    if target is not None:
        if str(target) not in _enemy_entity_ids(state):
            return False, f"target {target!r} not a living enemy"
    elif _card_needs_target(card) and _enemy_entity_ids(state):
        return False, "card needs target but none provided"

    return True, "ok"


def _validate_end_turn(state: dict) -> tuple[bool, str]:
    state_type = str(state.get("state_type") or "").lower()
    if state_type not in combat.COMBAT_STATE_TYPES:
        return False, f"end_turn invalid on state_type={state_type}"
    battle = state.get("battle") or {}
    if battle.get("is_play_phase") is False:
        return False, "not play phase"
    return True, "ok"


def _validate_hand_select(state: dict, action: dict) -> tuple[bool, str]:
    if str(state.get("state_type") or "").lower() != "hand_select":
        return False, "not hand_select"
    hs = state.get("hand_select") or {}
    name = str(action.get("action") or "")

    if name == "combat_confirm_selection":
        if hs.get("can_confirm"):
            return True, "ok"
        if hs.get("selected_cards"):
            return True, "ok"
        return False, "cannot confirm yet"

    if name == "combat_select_card":
        cards = hs.get("cards") or (state.get("player") or {}).get("hand") or []
        card_index = int(action.get("card_index", -1))
        for list_idx, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            if card.get("can_select") is False:
                continue
            api_idx = int(card.get("index") if card.get("index") is not None else list_idx)
            if api_idx == card_index:
                return True, "ok"
        return False, f"combat_select_card index {card_index} not selectable"

    return False, f"unexpected action in hand_select: {name}"


def _validate_card_select(state: dict, action: dict) -> tuple[bool, str]:
    if not is_card_select_active(state):
        return False, "card_select not active"
    screen = get_card_select_screen(state) or {}
    name = str(action.get("action") or "")

    if name == "confirm_selection":
        from sts2_agent.card_select import api_ready_to_confirm

        required = effective_required_count(screen)
        selected = effective_selected_indices(screen)
        if api_ready_to_confirm(screen, selected, required):
            return True, "ok"
        return False, f"need {required} selections, have {len(selected)}"

    if name == "select_card":
        cards = screen.get("cards") or []
        idx = int(action.get("index", -1))
        for list_idx, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            grid_idx = int(card.get("index") if card.get("index") is not None else list_idx)
            if grid_idx == idx:
                return True, "ok"
        return False, f"select_card index {idx} not in grid"

    return False, f"unexpected action in card_select: {name}"


def _validate_map(state: dict, action: dict) -> tuple[bool, str]:
    if str(state.get("state_type") or "").lower() != "map":
        return False, "not map screen"
    choices = extract_map_choices(state)
    if not choices:
        return False, "no map choices"
    idx = int(action.get("index", -1))
    for fallback, option in enumerate(choices):
        if map_choice_index(option, fallback) == idx:
            return True, "ok"
    return False, f"map index {idx} not available"


def _validate_rewards(state: dict, action: dict) -> tuple[bool, str]:
    state_type = str(state.get("state_type") or "").lower()
    name = str(action.get("action") or "")

    if state_type == "rewards":
        if name == "proceed":
            if rewards_can_proceed(state) and not extract_reward_items(state):
                return True, "ok"
            return False, "cannot proceed yet"
        if name == "claim_reward":
            from sts2_agent.rewards import get_rewards_flow

            idx = int(action.get("index", -1))
            if idx in get_rewards_flow().declined_card_reward_indices:
                return False, f"card reward index {idx} declined"
            items = extract_reward_items(state)
            for fallback, item in enumerate(items):
                if reward_item_index(item, fallback) == idx:
                    return True, "ok"
            return False, f"reward index {idx} not claimable"
        if name == "discard_potion":
            return _validate_potion_slot(state, action)
        return False, f"unexpected rewards action: {name}"

    if state_type == "card_reward":
        if name == "select_card_reward":
            cards = extract_card_reward_cards(state)
            idx = int(action.get("card_index", -1))
            for fallback, card in enumerate(cards):
                if card_reward_index(card, fallback) == idx:
                    return True, "ok"
            return False, f"card_reward index {idx} not offered"
        if name == "skip_card_reward":
            if card_reward_can_skip(state):
                return True, "ok"
            return False, "cannot skip card reward"
        return False, f"unexpected card_reward action: {name}"

    if state_type == "treasure":
        if name == "proceed":
            if treasure_can_proceed(state) and not extract_treasure_relics(state):
                return True, "ok"
            return False, "cannot proceed from treasure yet"
        if name == "claim_treasure_relic":
            relics = extract_treasure_relics(state)
            if not relics:
                return False, "no treasure relics to claim"
            idx = int(action.get("index", -1))
            for fallback, relic in enumerate(relics):
                grid_idx = int(relic.get("index") if relic.get("index") is not None else fallback)
                if grid_idx == idx:
                    return True, "ok"
            return False, f"treasure relic index {idx} not offered"

    return False, f"action {name} invalid for state_type={state_type}"


def _validate_rest(state: dict, action: dict) -> tuple[bool, str]:
    if str(state.get("state_type") or "").lower() != "rest_site":
        return False, "not rest_site"
    name = str(action.get("action") or "")
    if name == "proceed":
        if rest_can_proceed(state):
            return True, "ok"
        return False, "cannot proceed from rest"
    if name == "choose_rest_option":
        idx = int(action.get("index", -1))
        options = extract_rest_options(state)
        for fallback, option in enumerate(options):
            if rest_option_index(option, fallback) == idx:
                return True, "ok"
        return False, f"rest option {idx} not enabled"
    return False, f"unexpected rest action: {name}"


def _validate_shop(state: dict, action: dict) -> tuple[bool, str]:
    if str(state.get("state_type") or "").lower() not in ("shop", "fake_merchant"):
        return False, "not shop"
    name = str(action.get("action") or "")
    if name == "proceed":
        screen = get_shop_screen(state) or {}
        if screen.get("can_proceed") and not extract_shop_items(state):
            return True, "ok"
        return False, "cannot proceed from shop"
    if name == "shop_purchase":
        idx = int(action.get("index", -1))
        for item in extract_shop_items(state):
            if int(item.get("index", -1)) == idx:
                return True, "ok"
        return False, f"shop index {idx} not purchasable"
    return False, f"unexpected shop action: {name}"


def _validate_event(state: dict, action: dict) -> tuple[bool, str]:
    if str(state.get("state_type") or "").lower() != "event":
        return False, "not event"
    name = str(action.get("action") or "")

    if name == "advance_dialogue":
        if event_in_dialogue(state) or not extract_event_options(state):
            return True, "ok"
        return False, "event options available - advance_dialogue invalid"

    if name == "choose_event_option":
        if event_has_proceed_option(state):
            return False, "proceed option expected, not choose_event_option"
        idx = int(action.get("index", -1))
        for fallback, option in enumerate(extract_event_options(state)):
            if option.get("is_locked"):
                continue
            if event_option_index(option, fallback) == idx:
                return True, "ok"
        return False, f"event option {idx} not choosable"

    if name == "proceed":
        return True, "ok"

    return False, f"unexpected event action: {name}"


def _validate_potion_slot(state: dict, action: dict) -> tuple[bool, str]:
    slot = int(action.get("slot", -1))
    potions = (state.get("player") or {}).get("potions") or []
    if slot < 0 or slot >= len(potions):
        return False, f"potion slot {slot} out of range"
    potion = potions[slot]
    if not potion or (isinstance(potion, dict) and not potion.get("name")):
        return False, f"potion slot {slot} empty"
    target = action.get("target")
    if target is not None and str(target) not in _enemy_entity_ids(state):
        if _enemy_entity_ids(state):
            return False, f"potion target {target!r} not a living enemy"
    return True, "ok"


def normalize_policy_action(state: dict, action: dict) -> dict:
    """Map training-era action names to live API actions."""
    state_type = str(state.get("state_type") or "").lower()
    name = str(action.get("action") or "")
    if state_type == "card_reward" and name == "proceed":
        if card_reward_can_skip(state):
            return {"action": "skip_card_reward"}
    return action


def validate_policy_action(state: dict, action: dict | None) -> tuple[bool, str]:
    """Return (is_valid, reason) for a decoded policy action on this state."""
    if not action or not isinstance(action, dict):
        return False, "empty action"
    name = str(action.get("action") or "").strip()
    if not name:
        return False, "missing action name"

    if name == "play_card":
        return _validate_play_card(state, action)
    if name == "end_turn":
        return _validate_end_turn(state)
    if name in ("use_potion", "discard_potion"):
        state_type = str(state.get("state_type") or "").lower()
        if state_type not in combat.COMBAT_STATE_TYPES:
            return False, f"{name} invalid on state_type={state_type}"
        return _validate_potion_slot(state, action)

    if name in ("combat_select_card", "combat_confirm_selection"):
        return _validate_hand_select(state, action)
    if name in ("select_card", "confirm_selection"):
        return _validate_card_select(state, action)

    if name == "choose_map_node":
        return _validate_map(state, action)

    if name in (
        "claim_reward",
        "select_card_reward",
        "skip_card_reward",
        "claim_treasure_relic",
        "proceed",
    ):
        return _validate_rewards(state, action)

    if name == "choose_rest_option":
        return _validate_rest(state, action)

    if name == "shop_purchase":
        return _validate_shop(state, action)

    if name in ("advance_dialogue", "choose_event_option"):
        return _validate_event(state, action)

    if name == "proceed":
        state_type = str(state.get("state_type") or "").lower()
        if state_type == "rest_site":
            return _validate_rest(state, action)
        if state_type in ("shop", "fake_merchant"):
            return _validate_shop(state, action)
        if state_type in ("rewards", "card_reward", "treasure"):
            return _validate_rewards(state, action)
        return False, f"proceed invalid on state_type={state_type}"

    return False, f"unhandled action type: {name}"
