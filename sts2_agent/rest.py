"""Rest site decisions using scorer.

Training data: rest/smith choices recorded via data_pipeline (main.py).
Heal (+5) and smith (+3) immediate rewards applied in data_pipeline.
"""

from __future__ import annotations

from sts2_agent.data_pipeline import record_handler_decision
from sts2_agent.knowledge import get_knowledge
from sts2_agent.scorer import deck_cards, score_rest_option
from sts2_agent.state_parse import (
    extract_rest_options,
    rest_can_proceed,
    rest_option_index,
)


def record_training(state: dict, action: dict | None, reasoning: list[str]) -> None:
    record_handler_decision(state, action, reasoning, handler="rest")


def decide_rest_site(state: dict) -> tuple[dict | None, list[str]]:
    """
    Rest site flow (STS2MCP):
    1. choose_rest_option while options are available
    2. card_select overlay for smith (handled in agent before this runs)
    3. proceed only when rest_site.can_proceed is true
    """
    options = extract_rest_options(state)

    if rest_can_proceed(state) and not options:
        return {"action": "proceed"}, ["rest site done - proceed to map"]

    if not options:
        return None, ["rest site - waiting (no enabled options, cannot proceed yet)"]

    player = state.get("player") or {}
    hp = int(player.get("hp") or 0)
    max_hp = int(player.get("max_hp") or 1)
    hp_ratio = hp / max_hp if max_hp else 1.0
    kb = get_knowledge()
    deck = deck_cards(state)

    scored: list[tuple[int, float, list[str]]] = []
    all_reasons = [f"rest site: HP {hp}/{max_hp} ({hp_ratio:.0%})"]

    for list_idx, option in enumerate(options):
        api_index = rest_option_index(option, list_idx)
        result = score_rest_option(option, hp_ratio=hp_ratio, deck=deck, kb=kb)
        scored.append((api_index, result.score, result.reasons))
        all_reasons.append(
            f"  option[{api_index}]: {result.score:.1f} - {'; '.join(result.reasons)}"
        )

    scored.sort(key=lambda x: x[1], reverse=True)
    best_index, best_score, _best_reasons = scored[0]
    all_reasons.append(f"choose_rest_option index {best_index} (score {best_score:.1f})")
    return {"action": "choose_rest_option", "index": best_index}, all_reasons
