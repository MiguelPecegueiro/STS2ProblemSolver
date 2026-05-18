"""Extract screen-specific data from STS2MCP game state JSON."""

from __future__ import annotations

import re
from typing import Any


def extract_map_choices(state: dict) -> list[dict]:
    """
    Return choosable next map nodes (`map.next_options`).

    The API expects `choose_map_node.index` to be the index from this list,
    NOT an index into `map.nodes` (the full DAG).
    """
    map_data = state.get("map")
    if not isinstance(map_data, dict):
        return []

    next_options = map_data.get("next_options")
    if isinstance(next_options, list) and next_options:
        return [n for n in next_options if isinstance(n, dict)]

    # Fallbacks for older/alternate payloads
    for key in ("choices", "options", "available"):
        val = map_data.get(key)
        if isinstance(val, list):
            return [n for n in val if isinstance(n, dict)]
    return []


def map_choice_index(option: dict, fallback: int) -> int:
    if option.get("index") is not None:
        return int(option["index"])
    return fallback


def map_choice_room_type(option: dict) -> str:
    return str(option.get("type") or option.get("room_type") or option.get("label") or "")


def extract_reward_items(state: dict) -> list[dict]:
    rewards = state.get("rewards")
    if isinstance(rewards, dict):
        items = rewards.get("items")
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict) and not i.get("claimed")]
    if isinstance(rewards, list):
        return [i for i in rewards if isinstance(i, dict) and not i.get("claimed")]
    return []


def rewards_can_proceed(state: dict) -> bool:
    screen = state.get("rewards")
    if isinstance(screen, dict):
        return bool(screen.get("can_proceed", True))
    return True


def treasure_can_proceed(state: dict) -> bool:
    screen = state.get("treasure")
    if isinstance(screen, dict):
        return bool(screen.get("can_proceed"))
    return False


def treasure_is_opening(state: dict) -> bool:
    """Chest still animating - poll again before claim/proceed."""
    screen = state.get("treasure")
    if not isinstance(screen, dict):
        return False
    if screen.get("message"):
        return True
    return not screen.get("relics") and not screen.get("can_proceed")


def extract_treasure_relics(state: dict) -> list[dict]:
    """Unclaimed relic options on the treasure screen."""
    treasure = state.get("treasure")
    raw: list = []
    if isinstance(treasure, dict):
        for key in ("relics", "options", "items"):
            val = treasure.get(key)
            if isinstance(val, list):
                raw = val
                break
    if not raw:
        for key in ("treasure", "relics", "relic_options", "choices"):
            val = state.get(key)
            if isinstance(val, list):
                raw = val
                break
    return [
        r
        for r in raw
        if isinstance(r, dict) and not r.get("claimed") and r.get("name")
    ]


def extract_card_reward_cards(state: dict) -> list[dict]:
    screen = state.get("card_reward")
    if isinstance(screen, dict):
        cards = screen.get("cards")
        if isinstance(cards, list):
            return [c for c in cards if isinstance(c, dict)]

    for key in ("card_rewards", "cards", "choices", "options"):
        val = state.get(key)
        if isinstance(val, list):
            return [c for c in val if isinstance(c, dict)]
    return []


def card_reward_name_for_index(state: dict, card_index: int) -> str | None:
    """Resolve API card_index on card_reward screen to a display name."""
    try:
        target = int(card_index)
    except (TypeError, ValueError):
        return None
    for fallback, card in enumerate(extract_card_reward_cards(state)):
        if card_reward_index(card, fallback) == target:
            name = str(card.get("name") or "").strip()
            cid = str(card.get("id") or "").strip()
            return name or (cid.upper().replace(" ", "_") if cid else None)
    return None


def extract_card_reward_offered(state: dict) -> list[str]:
    """Card names (or ids) for every option on the reward screen."""
    offered: list[str] = []
    for card in extract_card_reward_cards(state):
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or "").strip()
        cid = str(card.get("id") or "").strip()
        label = name
        if not label and cid:
            label = cid.upper().replace(" ", "_")
        if label:
            offered.append(label)
    return offered


def card_reward_can_skip(state: dict) -> bool:
    screen = state.get("card_reward")
    if isinstance(screen, dict):
        return bool(screen.get("can_skip", True))
    return True


def card_reward_index(card: dict, fallback: int) -> int:
    if card.get("index") is not None:
        return int(card["index"])
    return fallback


def reward_item_index(item: dict, fallback: int) -> int:
    if item.get("index") is not None:
        return int(item["index"])
    return fallback


def extract_rest_options(state: dict) -> list[dict]:
    """
    Enabled rest-site choices from `rest_site.options` only.

    Do not read top-level `options` / `choices` - those belong to other screens
    and may be empty while `rest_site.options` still has Rest / Smith buttons.
    """
    rest = state.get("rest_site")
    if not isinstance(rest, dict):
        return []
    raw = rest.get("options")
    if not isinstance(raw, list):
        return []
    enabled: list[dict] = []
    for list_idx, option in enumerate(raw):
        if not isinstance(option, dict):
            continue
        if option.get("is_enabled") is False:
            continue
        enabled.append(option)
    return enabled


def rest_option_index(option: dict, fallback: int) -> int:
    if option.get("index") is not None:
        return int(option["index"])
    return fallback


def rest_can_proceed(state: dict) -> bool:
    rest = state.get("rest_site")
    if isinstance(rest, dict):
        return bool(rest.get("can_proceed"))
    return False


def get_card_select_screen(state: dict) -> dict | None:
    screen = state.get("card_select")
    return screen if isinstance(screen, dict) else None


def is_card_select_active(state: dict) -> bool:
    if str(state.get("state_type") or "").lower() == "card_select":
        return True
    screen = get_card_select_screen(state)
    if not screen:
        return False
    if screen.get("cards"):
        return True
    if screen.get("can_confirm") or screen.get("preview_showing"):
        return True
    return False


def _selection_count_from_prompt(prompt: str) -> int | None:
    if not prompt:
        return None
    for pattern in (
        # Choose 2 Common Cards to Add…
        r"(?:choose|select|pick)\s+(\d+)\s+(?:\w+\s+)*cards?\b",
        r"(?:choose|select|pick)\s+(\d+)\s+cards?\b",
        r"\b(\d+)\s+(?:common|uncommon|rare|basic|starter|colorless)\s+cards?\b",
        r"\b(\d+)\s+cards?\s+to\s+(?:transform|upgrade|remove|exhaust|add)",
        r"(?:transform|upgrade|remove|exhaust|add)\s+(\d+)\s+cards?\b",
    ):
        match = re.search(pattern, prompt, re.I)
        if match:
            return max(1, int(match.group(1)))
    return None


def card_select_is_transform(screen: dict) -> bool:
    screen_type = str(screen.get("screen_type") or "").lower()
    prompt = str(screen.get("prompt") or "").lower()
    return screen_type == "transform" or "transform" in prompt


def card_select_overlay_key(screen: dict) -> str:
    screen_type = str(screen.get("screen_type") or "")
    prompt = str(screen.get("prompt") or "")
    prompt_l = prompt.lower()

    if card_select_is_transform(screen):
        # Batch pick-N (Morphic Grove) must not flip to transform:1 when the prompt
        # changes during preview - use required_count (API field + prompt parsing).
        need = card_select_required_count(screen)
        if need > 1:
            return "transform:multi"
        return "transform:single"

    if card_select_single_pick_confirm(screen):
        return screen_type

    cards = screen.get("cards") or []
    card_ids: list[str] = []
    for card in cards[:40]:
        if isinstance(card, dict):
            card_ids.append(str(card.get("id") or card.get("name") or ""))
    return "|".join([screen_type, ",".join(sorted(card_ids))])


def card_select_single_pick_confirm(screen: dict) -> bool:
    if card_select_required_count(screen) > 1:
        return False
    screen_type = str(screen.get("screen_type") or "").lower()
    prompt = str(screen.get("prompt") or "").lower()
    if screen_type in ("transform", "upgrade", "smith", "select"):
        return True
    if any(k in screen_type for k in ("enchant", "upgrade", "smith", "transform")):
        return True
    if any(k in prompt for k in ("enchant", "upgrade", "transform", "smith", "remove")):
        return True
    return False


def card_select_required_count(screen: dict) -> int:
    for key in ("required_selections", "required_count", "selection_count_required"):
        raw = screen.get(key)
        if raw is not None:
            try:
                return max(1, int(raw))
            except (TypeError, ValueError):
                pass

    prompt = str(screen.get("prompt") or "")
    from_prompt = _selection_count_from_prompt(prompt)
    if from_prompt is not None:
        return from_prompt

    screen_type = str(screen.get("screen_type") or "").lower()
    if screen_type in ("upgrade", "smith", "simple_select", "choose"):
        return 1
    if screen_type in ("multiselect", "multi_select"):
        match = re.search(r"\b(\d+)\b", prompt)
        if match:
            return max(2, int(match.group(1)))
        return 2
    return 1


def card_select_is_immediate(screen: dict) -> bool:
    screen_type = str(screen.get("screen_type") or "").lower()
    if screen_type == "choose":
        return True
    required = card_select_required_count(screen)
    return required <= 1 and screen_type not in ("transform", "select", "upgrade")


def card_select_selected_indices(screen: dict) -> set[int]:
    selected: set[int] = set()

    for entry in screen.get("selected_cards") or []:
        if isinstance(entry, dict) and entry.get("index") is not None:
            selected.add(int(entry["index"]))
        elif isinstance(entry, int):
            selected.add(int(entry))

    for raw in screen.get("selected_indices") or []:
        try:
            selected.add(int(raw))
        except (TypeError, ValueError):
            pass

    for key in ("selected_card_indices", "selected_indexes"):
        for raw in screen.get(key) or []:
            try:
                selected.add(int(raw))
            except (TypeError, ValueError):
                pass

    for card in screen.get("cards") or []:
        if not isinstance(card, dict):
            continue
        if card.get("is_selected") or card.get("selected"):
            idx = card.get("index")
            if idx is not None:
                selected.add(int(idx))

    return selected


def card_select_grid_index(card: dict, list_idx: int) -> int:
    if card.get("index") is not None:
        return int(card["index"])
    return int(list_idx)


def get_shop_screen(state: dict) -> dict | None:
    shop = state.get("shop")
    if isinstance(shop, dict):
        return shop
    merchant = state.get("fake_merchant")
    if isinstance(merchant, dict) and isinstance(merchant.get("shop"), dict):
        return merchant["shop"]
    return None


def shop_item_name(item: dict) -> str:
    return str(
        item.get("card_name")
        or item.get("relic_name")
        or item.get("potion_name")
        or item.get("name")
        or ""
    ).strip()


def shop_item_category(item: dict) -> str:
    return str(item.get("category") or item.get("type") or "").lower()


def shop_item_index(item: dict, fallback: int) -> int:
    if item.get("index") is not None:
        return int(item["index"])
    return fallback


def shop_item_price(item: dict) -> int:
    for key in ("price", "cost", "gold"):
        if item.get(key) is not None:
            return int(item[key])
    return 9999


def extract_shop_items(state: dict) -> list[dict]:
    """
    Purchasable shop rows from `shop.items` only.

    Skips sold-out slots, unaffordable rows, and empty placeholders so we do not
    spam `shop_purchase` on index 0 after a relic slot is empty.
    """
    screen = get_shop_screen(state)
    if not screen:
        return []
    raw = screen.get("items")
    if not isinstance(raw, list):
        return []
    player = state.get("player") or {}
    gold = int(player.get("gold") or 0)
    purchasable: list[dict] = []
    for list_idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        if item.get("is_stocked") is False:
            continue
        category = shop_item_category(item)
        if not category and not shop_item_name(item):
            continue
        price = shop_item_price(item)
        if item.get("can_afford") is False or price > gold:
            continue
        purchasable.append(item)
    return purchasable


def shop_can_proceed(state: dict) -> bool:
    screen = get_shop_screen(state)
    if isinstance(screen, dict):
        return bool(screen.get("can_proceed"))
    return False


def shop_inventory_ready(state: dict) -> bool:
    screen = get_shop_screen(state)
    if not isinstance(screen, dict):
        return False
    if screen.get("error"):
        return False
    return isinstance(screen.get("items"), list)


def cheapest_stocked_price(state: dict) -> int | None:
    screen = get_shop_screen(state)
    if not isinstance(screen, dict):
        return None
    raw = screen.get("items")
    if not isinstance(raw, list):
        return None
    prices: list[int] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if item.get("is_stocked") is False:
            continue
        if not shop_item_category(item) and not shop_item_name(item):
            continue
        prices.append(shop_item_price(item))
    return min(prices) if prices else None


def get_event_screen(state: dict) -> dict | None:
    event = state.get("event")
    return event if isinstance(event, dict) else None


def event_option_label(option: dict) -> str:
    return str(
        option.get("title")
        or option.get("text")
        or option.get("label")
        or option.get("description")
        or option.get("name")
        or ""
    ).strip()


def event_option_index(option: dict, fallback: int) -> int:
    if option.get("index") is not None:
        return int(option["index"])
    return fallback


def extract_event_options(state: dict) -> list[dict]:
    """
    Choosable event options from `event.options` only.

    Skips locked rows. Includes Proceed buttons (`is_proceed`).
    """
    screen = get_event_screen(state)
    if not screen:
        return []
    raw = screen.get("options")
    if not isinstance(raw, list):
        return []
    options: list[dict] = []
    for list_idx, option in enumerate(raw):
        if not isinstance(option, dict):
            continue
        if option.get("is_locked"):
            continue
        options.append(option)
    return options


def _is_real_event_choice(option: dict) -> bool:
    if option.get("is_proceed"):
        return False
    label = event_option_label(option).lower()
    if any(k in label for k in ("proceed", "continue", "leave", "exit", "done", "next")):
        return False
    return True


def event_has_choosable_options(state: dict) -> bool:
    """True when real (non-proceed) event options exist on screen."""
    for option in extract_event_options(state):
        if _is_real_event_choice(option):
            return True
    return False


def event_has_unchosen_choices(state: dict) -> bool:
    """True when the player still needs to pick a real event option."""
    for option in extract_event_options(state):
        if option.get("was_chosen") or option.get("is_locked"):
            continue
        if _is_real_event_choice(option):
            return True
    return False


def event_in_dialogue(state: dict) -> bool:
    """True when the agent should call advance_dialogue before choosing options."""
    if state.get("awaiting_dialogue") or state.get("dialogue_active"):
        return True
    screen = get_event_screen(state)
    if not screen:
        return False
    if screen.get("in_dialogue"):
        return True
    # Heuristic: narrative showing but no real choices yet (common on Ancients)
    if event_has_unchosen_choices(state):
        return False
    options = extract_event_options(state)
    if event_has_proceed_option(state) and not event_has_unchosen_choices(state):
        return False
    if options and not event_has_proceed_option(state):
        return False
    if screen.get("body") or screen.get("dialogue") or screen.get("is_ancient"):
        return True
    return False


def event_has_proceed_option(state: dict) -> bool:
    for option in extract_event_options(state):
        if option.get("is_proceed"):
            return True
        label = event_option_label(option).lower()
        if any(
            word in label
            for word in ("proceed", "continue", "leave", "exit", "done", "next", "ok")
        ):
            return True
    return False


def living_enemies(battle: dict | None) -> list[dict]:
    """Enemies with HP > 0."""
    if not isinstance(battle, dict):
        return []
    living: list[dict] = []
    for enemy in battle.get("enemies") or []:
        if isinstance(enemy, dict) and int(enemy.get("hp") or 0) > 0:
            living.append(enemy)
    return living


def hand_has_playable_cards(player: dict | None) -> bool:
    """True if any hand card is not explicitly marked unplayable."""
    if not isinstance(player, dict):
        return False
    for card in player.get("hand") or []:
        if not isinstance(card, dict):
            continue
        if card.get("can_play") is False or card.get("playable") is False:
            continue
        return True
    return False


def combat_awaiting_enemies(state: dict) -> bool:
    """
    True during play phase when targets are temporarily missing (e.g. Phrog split).

    The API may report zero living enemies while spawn/split animations run, but the
    player can still have energy and cards - do not end_turn in that window.
    """
    battle = state.get("battle") or {}
    if battle.get("is_play_phase") is False:
        return False
    if str(battle.get("turn") or "").lower() != "player":
        return False
    if living_enemies(battle):
        return False

    player = state.get("player") or {}
    energy = int(player.get("energy") or player.get("current_energy") or 0)
    if energy > 0:
        return True
    return hand_has_playable_cards(player)
