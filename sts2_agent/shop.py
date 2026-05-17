"""Shop purchase heuristics.

Training data: purchase/proceed decisions recorded via data_pipeline (main.py).
"""

from __future__ import annotations

from sts2_agent.data_pipeline import record_handler_decision
from sts2_agent.knowledge import _normalize_name, get_knowledge
from sts2_agent.scorer import score_shop_item
from sts2_agent.state_parse import (
    cheapest_stocked_price,
    extract_shop_items,
    shop_can_proceed,
    shop_inventory_ready,
    shop_item_index,
    shop_item_price,
)


def record_training(state: dict, action: dict | None, reasoning: list[str]) -> None:
    record_handler_decision(state, action, reasoning, handler="shop")


def decide_shop(state: dict) -> tuple[dict | None, list[str]]:
    """
    Shop flow (STS2MCP):
    1. shop_purchase using each item's `index` while stocked, affordable, worthwhile
    2. proceed to leave - especially when broke or nothing left to buy

    Do not wait on can_proceed when broke; the Back button may still need proceed
  even if can_proceed is false in API state.
    """
    reasons: list[str] = []

    if not shop_inventory_ready(state):
        return None, ["shop inventory not ready - wait"]

    player = state.get("player") or {}
    gold = int(player.get("gold") or 0)
    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    hp_ratio = hp / max_hp if max_hp else 1.0
    can_leave = shop_can_proceed(state)
    cheapest = cheapest_stocked_price(state)
    reasons.append(
        f"shop: {gold} gold, HP {hp}/{max_hp} ({hp_ratio:.0%}), "
        f"can_proceed={can_leave}, cheapest_stocked={cheapest}"
    )

    owned_ids = {
        str(r.get("id") or "").lower()
        for r in (player.get("relics") or [])
        if isinstance(r, dict)
    }
    owned_names = {
        _normalize_name(str(r.get("name") or ""))
        for r in (player.get("relics") or [])
        if isinstance(r, dict)
    }

    items = extract_shop_items(state)

    if not items:
        return _leave_shop(
            reasons,
            f"nothing affordable ({gold}g left, cheapest item {cheapest}g)",
        )

    if cheapest is not None and gold < cheapest:
        return _leave_shop(
            reasons,
            f"cannot afford any stocked item (have {gold}g, cheapest {cheapest}g)",
        )

    kb = get_knowledge()
    scored: list[tuple[int, float, int, list[str]]] = []
    for list_idx, item in enumerate(items):
        api_index = shop_item_index(item, list_idx)
        price = shop_item_price(item)
        result = score_shop_item(
            item,
            state=state,
            hp_ratio=hp_ratio,
            gold=gold,
            owned_relic_ids=owned_ids,
            owned_relic_names=owned_names,
            kb=kb,
        )
        scored.append((api_index, result.score, price, result.reasons))
        reasons.append(f"  [{api_index}] {result.score:.1f} - {'; '.join(result.reasons)}")

    def _sort_key(entry: tuple[int, float, int, list[str]]) -> tuple[float, int]:
        api_index, score, price, _detail = entry
        item = next(
            (i for i in items if shop_item_index(i, 0) == api_index),
            {},
        )
        category = str(item.get("category") or "").lower()
        relic_bonus = 5.0 if category == "relic" else 0.0
        return (score + relic_bonus, -price)

    scored.sort(key=_sort_key, reverse=True)
    best_index, best_score, best_price, _best_detail = scored[0]

    min_score = 35.0
    gold_reserve = 50 if hp_ratio >= 0.5 else 20

    if best_price > gold:
        return _leave_shop(reasons, f"best pick costs {best_price}g but only have {gold}g")

    if best_score < min_score:
        return _leave_shop(reasons, f"best offer score {best_score:.1f} < {min_score}")

    if gold - best_price < gold_reserve and best_score < 80:
        return _leave_shop(
            reasons,
            f"keeping gold reserve ({gold_reserve}g) - best {best_score:.1f}",
        )

    reasons.append(f"shop_purchase index {best_index} (score {best_score:.1f}, {best_price}g)")
    return {"action": "shop_purchase", "index": best_index}, reasons


def _leave_shop(reasons: list[str], why: str) -> tuple[dict, list[str]]:
    """Always try proceed to exit - do not wait on can_proceed."""
    return {"action": "proceed"}, reasons + [f"leave shop - {why}"]
