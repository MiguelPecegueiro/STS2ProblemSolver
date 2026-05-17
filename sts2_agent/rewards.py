"""Post-combat rewards, card picks, treasure."""

from __future__ import annotations

from sts2_agent.data_pipeline import record_handler_decision
from sts2_agent.knowledge import get_knowledge
from sts2_agent.potions import (
    get_potion_drop_tracker,
    potion_belt_full,
    score_offered_potion_reward,
    score_potion,
    worst_potion_slot,
)
from sts2_agent.scorer import ScoredOption, card_name, score_card_reward
from sts2_agent.state_parse import (
    card_reward_can_skip,
    card_reward_index,
    extract_card_reward_cards,
    extract_reward_items,
    extract_treasure_relics,
    reward_item_index,
    rewards_can_proceed,
    treasure_can_proceed,
    treasure_is_opening,
)


def record_training(state: dict, action: dict | None, reasoning: list[str]) -> None:
    record_handler_decision(state, action, reasoning, handler="rewards")


class RewardsFlow:
    """Handles multi-step potion claim (discard slot, then claim reward)."""

    def __init__(self) -> None:
        self.pending_potion_claim_index: int | None = None
        self.pending_card_reward_index: int | None = None
        self.declined_card_reward_indices: set[int] = set()

    def clear(self) -> None:
        self.pending_potion_claim_index = None
        self.pending_card_reward_index = None
        self.declined_card_reward_indices.clear()

    def clear_rewards_screen(self) -> None:
        """Reset per-screen state when leaving the post-combat rewards flow."""
        self.pending_potion_claim_index = None
        self.pending_card_reward_index = None
        self.declined_card_reward_indices.clear()


def _is_card_reward_item(item: dict) -> bool:
    return str(item.get("type") or "").lower() in ("card", "special_card")


def note_card_reward_claimed(state: dict, index: int) -> None:
    """Remember which rewards row opened the card-pick screen."""
    flow = get_rewards_flow()
    for fallback, item in enumerate(extract_reward_items(state)):
        if reward_item_index(item, fallback) == index and _is_card_reward_item(item):
            flow.pending_card_reward_index = index
            return


def note_card_reward_skipped() -> None:
    """After skip_card_reward, do not re-claim the same card reward row."""
    flow = get_rewards_flow()
    if flow.pending_card_reward_index is not None:
        flow.declined_card_reward_indices.add(flow.pending_card_reward_index)
    flow.pending_card_reward_index = None


def note_card_reward_selected() -> None:
    get_rewards_flow().pending_card_reward_index = None


def note_rewards_screen_done() -> None:
    get_rewards_flow().clear_rewards_screen()


_flow = RewardsFlow()


def get_rewards_flow() -> RewardsFlow:
    return _flow


def decide_card_reward(state: dict) -> tuple[dict, list[str]]:
    cards = extract_card_reward_cards(state)
    if not cards:
        return {"action": "skip_card_reward"}, ["no card_reward.cards - skip"]

    kb = get_knowledge()
    scored: list[tuple[int, int, ScoredOption]] = []
    for list_idx, card in enumerate(cards):
        api_index = card_reward_index(card, list_idx)
        result = score_card_reward(card, state, kb)
        scored.append((api_index, list_idx, result))

    scored.sort(key=lambda x: x[2].score, reverse=True)
    best_api_index, _, best = scored[0]
    reasons = [
        f"best card: {best.label} (score {best.score:.1f}, index {best_api_index})"
    ] + best.reasons
    for api_index, _, opt in scored[1:]:
        reasons.append(f"  alt [{api_index}] {opt.label}: {opt.score:.1f}")

    if best.score < -50 and card_reward_can_skip(state):
        return {"action": "skip_card_reward"}, reasons + ["score too low - skip"]
    return {
        "action": "select_card_reward",
        "card_index": best_api_index,
    }, reasons


def decide_rewards(state: dict) -> tuple[dict | None, list[str]]:
    flow = get_rewards_flow()
    player = state.get("player") or {}
    kb = get_knowledge()
    items = extract_reward_items(state)
    tracker = get_potion_drop_tracker()
    drop_chance = tracker.estimated_drop_chance(state)
    tracker.note_rewards_screen(state, items)

    if flow.pending_potion_claim_index is not None:
        idx = flow.pending_potion_claim_index
        flow.pending_potion_claim_index = None
        tracker.note_potion_taken()
        return {"action": "claim_reward", "index": idx}, [
            f"claim potion reward index {idx} after making belt room"
        ]

    if not items:
        flow.clear()
        if rewards_can_proceed(state):
            return {"action": "proceed"}, ["all rewards claimed - proceed to map"]
        return {"action": "proceed"}, ["no reward items visible - proceed"]

    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    hp_ratio = hp / max_hp if max_hp else 1.0
    belt_full = potion_belt_full(state)

    reasons: list[str] = [
        f"post-combat rewards (belt_full={belt_full}, est_potion_chance={drop_chance:.0%}):"
    ]

    claimable: list[tuple[dict, float, list[str]]] = []

    for item in items:
        api_index = reward_item_index(item, 0)
        item_type = str(item.get("type") or "").lower()
        item_reasons: list[str] = []

        if api_index in flow.declined_card_reward_indices:
            reasons.append(f"  [{api_index}] SKIP - card reward declined earlier")
            continue

        if item_type == "potion":
            offered_score = score_offered_potion_reward(item, state, kb)
            name = item.get("potion_name") or item.get("description") or "potion"
            item_reasons.append(f"potion {name} value {offered_score:.1f}")

            if belt_full:
                worst_slot, worst_score = worst_potion_slot(player, kb)
                item_reasons.append(
                    f"belt full - worst slot {worst_slot} ({worst_score:.1f})"
                )
                if offered_score <= worst_score + 3:
                    item_reasons.append("skip - not better than worst belt potion")
                    reasons.append(
                        f"  [{api_index}] SKIP potion - {name} ({'; '.join(item_reasons)})"
                    )
                    continue
                # Need to discard before claim
                flow.pending_potion_claim_index = api_index
                reasons.append(
                    f"  [{api_index}] potion upgrade - discard slot {worst_slot} then claim"
                )
                return {"action": "discard_potion", "slot": worst_slot}, reasons + item_reasons

            if tracker.should_deprioritize_potion_offer(state, is_elite=False):
                offered_score -= 25
                item_reasons.append("low drop chance - deprioritize")

            claimable.append((item, offered_score, item_reasons))
            reasons.append(
                f"  [{api_index}] potion score={offered_score:.1f} - {'; '.join(item_reasons)}"
            )
            continue

        score = _score_reward_item(item, state, hp_ratio, drop_chance, kb)
        claimable.append((item, score, [f"type={item_type}"]))
        reasons.append(
            f"  [{api_index}] type={item_type} score={score:.1f} - "
            f"{str(item.get('description', ''))[:50]}"
        )

    if not claimable:
        flow.pending_potion_claim_index = None
        tracker.note_potion_skipped()
        if rewards_can_proceed(state):
            return {"action": "proceed"}, reasons + [
                "nothing left to claim - proceed"
            ]
        return {"action": "proceed"}, reasons + [
            "only skipped potion(s) left - proceed"
        ]

    # Claim highest index first (STS2MCP convention)
    claimable.sort(key=lambda x: reward_item_index(x[0], 0), reverse=True)
    best_item, best_score, best_detail = max(claimable, key=lambda x: x[1])
    best_index = reward_item_index(best_item, 0)
    best_type = str(best_item.get("type") or "").lower()

    if best_type == "potion":
        tracker.note_potion_taken()
    elif belt_full is False and not any(
        str(i.get("type") or "").lower() == "potion" for i in items
    ):
        tracker.note_potion_skipped()

    reasons.append(f"claim_reward index {best_index} (type={best_type}, score {best_score:.1f})")
    if _is_card_reward_item(best_item):
        flow.pending_card_reward_index = best_index
    return {"action": "claim_reward", "index": best_index}, reasons + best_detail


def decide_treasure(state: dict) -> tuple[dict | None, list[str]]:
    if treasure_is_opening(state):
        return None, ["treasure chest opening - wait for next poll"]

    relics = extract_treasure_relics(state)
    if not relics:
        if treasure_can_proceed(state):
            return {"action": "proceed"}, ["treasure done - proceed to map"]
        return None, ["treasure - waiting for proceed to become available"]

    player = state.get("player") or {}
    owned = {str(r.get("name") or r).lower() for r in (player.get("relics") or [])}
    kb = get_knowledge()

    best_index, best_score = 0, float("-inf")
    reasons: list[str] = ["treasure relic options:"]

    for list_idx, relic in enumerate(relics):
        api_index = int(relic.get("index") if relic.get("index") is not None else list_idx)
        name = str(relic.get("name") if isinstance(relic, dict) else relic)
        if name.lower() in owned:
            reasons.append(f"  [{api_index}] {name}: owned - skip")
            continue
        codex = kb.lookup_relic(name)
        score = 50.0
        if codex:
            rarity = str(codex.get("rarity_key") or "").lower()
            if "rare" in rarity or "boss" in rarity:
                score += 30
        reasons.append(f"  [{api_index}] {name}: {score:.1f}")
        if score > best_score:
            best_score = score
            best_index = api_index

    if best_score <= float("-inf"):
        if treasure_can_proceed(state):
            return {"action": "proceed"}, reasons + ["no claimable relics - proceed"]
        return None, reasons + ["no claimable relics - waiting to proceed"]
    return {"action": "claim_treasure_relic", "index": best_index}, reasons + [
        f"claim relic index {best_index}"
    ]


def _score_reward_item(
    item: dict,
    state: dict,
    hp_ratio: float,
    drop_chance: float,
    kb,
) -> float:
    """Score a single rewards.items entry."""
    label = str(item.get("type") or "").lower()
    if label == "relic":
        return 95.0
    if label == "potion":
        base = score_offered_potion_reward(item, state, kb)
        if get_potion_drop_tracker().should_deprioritize_potion_offer(state, is_elite=False):
            return base - 20
        return base
    if label == "gold":
        amount = int(item.get("gold_amount") or item.get("amount") or 0)
        return 50.0 + min(amount, 100) / 5
    if label in ("card", "special_card"):
        return 65.0
    if label == "card_removal":
        return 60.0
    if "key" in label:
        return 85.0
    return 40.0
