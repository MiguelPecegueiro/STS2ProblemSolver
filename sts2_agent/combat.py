"""Combat decisions (scorer + knowledge)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

from sts2_agent.enemy_patterns import (
    assess_combat_debuff_pressure,
    enrich_incoming_damage,
    last_resolved_intents,
    record_enemy_intents_from_state,
)
from sts2_agent.knowledge import get_knowledge
from sts2_agent.pile_odds import format_pile_summary, next_turn_combat_estimates
from sts2_agent.scorer import (
    affordable_block_total,
    card_applies_vulnerable,
    card_name,
    enemy_has_vulnerable,
    estimate_hand_damage,
    find_lethal_target,
    pick_highest_threat_enemy,
    score_combat_play,
    smith_upgrade_priority,
    total_incoming_attack_damage,
)
from sts2_agent.state_parse import combat_awaiting_enemies, living_enemies

COMBAT_STATE_TYPES = frozenset({"monster", "elite", "boss"})

# combat_select_card sent but API did not mark the card selected (toggle / POST desync).
_hand_select_attempted: dict[str, set[int]] = {}

# When False, compendium still records intents but does not steer combat (see --no-compendium).
_compendium_decisions_enabled = True
_combat_solver_enabled = True


def compendium_decisions_enabled() -> bool:
    return _compendium_decisions_enabled


def set_compendium_decisions_enabled(enabled: bool) -> None:
    global _compendium_decisions_enabled
    _compendium_decisions_enabled = enabled


def set_combat_solver_enabled(enabled: bool) -> None:
    global _combat_solver_enabled
    _combat_solver_enabled = enabled
    from sts2_agent import combat_solver

    combat_solver.set_combat_solver_enabled(enabled)


def combat_solver_enabled() -> bool:
    return _combat_solver_enabled


def record_training(state: dict, action: dict | None, reasoning: list[str]) -> None:
    from sts2_agent.data_pipeline import record_handler_decision

    record_handler_decision(state, action, reasoning, handler="combat")


def decide_combat(state: dict) -> tuple[dict | None, list[str]]:
    # hand_select happens with is_play_phase false - handle before that guard.
    if state.get("state_type") == "hand_select":
        return _decide_hand_select(state)

    battle = state.get("battle") or {}
    if battle.get("is_play_phase") is False:
        return None, ["waiting - not play phase"]
    from sts2_agent.state_parse import is_player_combat_turn

    if not is_player_combat_turn(state):
        return None, ["waiting - not player turn"]

    kb = get_knowledge()
    player = state.get("player") or {}
    hand = player.get("hand") or []
    energy = int(player.get("energy") or player.get("current_energy") or 0)
    block = int(player.get("block") or 0)
    living = living_enemies(battle)
    floor = int((state.get("run") or {}).get("floor") or 0)
    reasons: list[str] = []

    record_enemy_intents_from_state(state)
    pile_summary = format_pile_summary(player)
    reasons.append(
        f"draw={pile_summary['draw_count']} top={pile_summary['draw_top'][:4]} "
        f"discard={pile_summary['discard_count']}"
    )

    if not living:
        if combat_awaiting_enemies(state):
            return None, [
                "waiting - no living enemies yet (split/spawn animation?)",
                f"energy={energy} hand={len(hand)}",
            ]
        return {"action": "end_turn"}, ["no living enemies - end turn"]

    incoming, incoming_reasons = total_incoming_attack_damage(living)
    reasons.extend(incoming_reasons)
    next_incoming = 0
    debuff_pressure = False

    if _compendium_decisions_enabled:
        kb_this, next_incoming, kb_reasons = enrich_incoming_damage(living, state)
        reasons.extend(kb_reasons)
        debuff_pressure, debuff_reasons = assess_combat_debuff_pressure(living, state)
        reasons.extend(debuff_reasons)
        for _eid, obs in last_resolved_intents(state).items():
            if obs.get("prediction_match") is False:
                reasons.append(
                    f"intent reconcile: predicted {obs.get('predicted_move') or obs.get('move_key')} "
                    f"got {obs.get('move_name') or obs.get('api_label')} "
                    f"dmg={obs.get('damage')} ({obs.get('source')})"
                )
        if incoming == 0 and kb_this > 0:
            reasons.append(
                f"compendium estimates {kb_this} dmg this turn - ignored "
                f"(live intents are non-attack); next_turn≈{next_incoming}"
            )
    else:
        reasons.append("compendium decisions off - live intents only")

    next_est = next_turn_combat_estimates(player, kb, energy=energy)
    reasons.extend(next_est.reasons)
    next_turn_risk = max(next_incoming - next_est.expected_block, 0)
    if _compendium_decisions_enabled:
        reasons.append(
            f"next_turn: incoming≈{next_incoming} est_block≈{next_est.expected_block} "
            f"risk={next_turn_risk}"
        )

    affordable_block = affordable_block_total(hand, energy, kb)
    effective_block = block + affordable_block
    needs_block_first = incoming > 0 and incoming > effective_block
    if _compendium_decisions_enabled and debuff_pressure:
        if incoming > 0 and incoming > effective_block:
            needs_block_first = True
            reasons.append("debuff pressure - prioritize block")
        elif incoming > 0:
            reasons.append("debuff pressure - elevated block priority")
            needs_block_first = needs_block_first or incoming > int(effective_block * 0.7)
        else:
            reasons.append(
                "debuff pressure - no attack incoming this turn; not forcing block-first"
            )
    reasons.append(
        f"this-turn incoming={incoming} block={block} affordable_block={affordable_block} "
        f"effective_block={effective_block} needs_block_first={needs_block_first}"
    )

    lethal_damage, lethal_est_reasons = estimate_hand_damage(hand, energy, living, kb)
    reasons.extend(lethal_est_reasons)
    lethal_target, lethal_target_reasons = find_lethal_target(living, lethal_damage)
    if lethal_target:
        reasons.extend(lethal_target_reasons)
        needs_block_first = False
        reasons.append("lethal available - skip block-first")

    playable: list[tuple[int, dict, float, list[str]]] = []
    for index, card in enumerate(hand):
        scored = score_combat_play(
            card,
            energy=energy,
            incoming_damage=incoming,
            current_block=block,
            enemies=living,
            kb=kb,
            floor=floor,
            lethal_damage=lethal_damage if lethal_target else None,
            next_turn_incoming=next_incoming,
            expected_block_next_turn=next_est.expected_block,
        )
        if scored.score > 0:
            playable.append((index, card, scored.score, scored.reasons))

    potion_action, pot_reasons = decide_combat_potion_from_context(
        state,
        player=player,
        living=living,
        incoming=incoming,
        block=block,
        lethal_target=lethal_target,
        lethal_damage=lethal_damage,
        has_playable_cards=bool(playable),
    )
    reasons.extend(pot_reasons)
    if potion_action:
        return potion_action, reasons

    if _combat_solver_enabled:
        try:
            from sts2_agent.combat_solver import try_solver_decide

            solved = try_solver_decide(state)
            if solved is not None:
                action, solver_reasons = solved
                return action, reasons + solver_reasons
        except Exception as exc:
            logger.warning("combat solver failed, using legacy combat: %s", exc)

    if not playable:
        return {"action": "end_turn"}, reasons + ["no playable cards - end turn"]

    if lethal_target:
        attack_plays = [p for p in playable if _is_attack_play(p[1], kb)]
        if attack_plays:
            attack_plays.sort(key=lambda x: x[2], reverse=True)
            index, card, _, card_reasons = attack_plays[0]
            action = _build_play_action(
                index, card, living, kb, prefer_entity_id=lethal_target.get("entity_id")
            )
            if action:
                return _finish_play(
                    action,
                    reasons,
                    playable,
                    index,
                    incoming,
                    needs_block_first,
                    "lethal - kill attacker",
                    card_reasons,
                    kb,
                )

    if not any(enemy_has_vulnerable(e) for e in living):
        vuln_setup = [p for p in playable if card_applies_vulnerable(p[1], kb)]
        if vuln_setup:
            vuln_setup.sort(key=lambda x: x[2], reverse=True)
            index, card, _, card_reasons = vuln_setup[0]
            action = _build_play_action(index, card, living, kb)
            if action:
                return _finish_play(
                    action,
                    reasons,
                    playable,
                    index,
                    incoming,
                    needs_block_first,
                    "apply Vulnerable before attacks (e.g. Bash)",
                    card_reasons,
                    kb,
                )

    safe_for_powers = incoming <= effective_block or incoming == 0
    if safe_for_powers:
        power_plays = [p for p in playable if _is_power_play(p[1], kb)]
        if power_plays:
            power_plays.sort(key=lambda x: x[2], reverse=True)
            index, card, _, card_reasons = power_plays[0]
            action = _build_play_action(index, card, living, kb)
            if action:
                return _finish_play(
                    action,
                    reasons,
                    playable,
                    index,
                    incoming,
                    needs_block_first,
                    "play power (safe turn)",
                    card_reasons,
                    kb,
                )

    if needs_block_first:
        block_plays = [p for p in playable if _is_pure_block_play(p[1], kb)]
        if block_plays:
            block_plays.sort(key=lambda x: x[2], reverse=True)
            index, card, _, card_reasons = block_plays[0]
            action = _build_play_action(index, card, living, kb)
            if action:
                return _finish_play(
                    action,
                    reasons,
                    playable,
                    index,
                    incoming,
                    needs_block_first,
                    f"block first (incoming {incoming} > effective block {effective_block})",
                    card_reasons,
                    kb,
                )

    debuff_plays = [p for p in playable if _is_debuff_play(p[1], kb)]
    if debuff_plays and energy >= 2 and not lethal_target:
        debuff_plays.sort(key=lambda x: x[2], reverse=True)
        index, card, _, card_reasons = debuff_plays[0]
        action = _build_play_action(index, card, living, kb)
        if action:
            return _finish_play(
                action,
                reasons,
                playable,
                index,
                incoming,
                needs_block_first,
                "apply debuff before damage",
                card_reasons,
                kb,
            )

    vuln_plays = [
        p
        for p in playable
        if _is_attack_play(p[1], kb) and any(enemy_has_vulnerable(e) for e in living)
    ]
    if vuln_plays:
        vuln_plays.sort(key=lambda x: x[2], reverse=True)
        index, card, _, card_reasons = vuln_plays[0]
        action = _build_play_action(index, card, living, kb, prefer_vulnerable=True)
        if action:
            return _finish_play(
                action,
                reasons,
                playable,
                index,
                incoming,
                needs_block_first,
                "damage vulnerable target",
                card_reasons,
                kb,
            )

    playable.sort(key=lambda x: x[2], reverse=True)
    index, card, _, card_reasons = playable[0]
    action = _build_play_action(index, card, living, kb)
    if action:
        return _finish_play(
            action,
            reasons,
            playable,
            index,
            incoming,
            needs_block_first,
            "highest score",
            card_reasons,
            kb,
        )

    return {"action": "end_turn"}, reasons + ["fallback end turn"]


def _play_comparison_log(
    playable: list[tuple[int, dict, float, list[str]]],
    chosen_index: int,
    *,
    incoming: int,
    needs_block_first: bool,
    path: str,
    kb,
) -> list[str]:
    """Explain incoming damage vs top attack/block scores for this decision."""
    lines = [
        f"play decision ({path}): incoming={incoming} needs_block_first={needs_block_first}"
    ]
    attacks = [p for p in playable if _is_attack_play(p[1], kb)]
    blocks = [p for p in playable if _is_pure_block_play(p[1], kb)]
    if attacks:
        idx, card, score, _ = max(attacks, key=lambda x: x[2])
        lines.append(
            f"  best attack: [{idx}] {card_name(card)} score={score:.0f}"
        )
    else:
        lines.append("  best attack: (none)")
    if blocks:
        idx, card, score, _ = max(blocks, key=lambda x: x[2])
        lines.append(
            f"  best pure block: [{idx}] {card_name(card)} score={score:.0f}"
        )
    else:
        lines.append("  best pure block: (none)")
    chosen = next((p for p in playable if p[0] == chosen_index), None)
    if chosen:
        _, card, score, _ = chosen
        kind = "attack" if _is_attack_play(card, kb) else (
            "block" if _is_pure_block_play(card, kb) else "other"
        )
        lines.append(
            f"  chosen: [{chosen_index}] {card_name(card)} ({kind}) score={score:.0f}"
        )
        if incoming <= 0 and kind == "block" and attacks:
            atk_score = max(attacks, key=lambda x: x[2])[2]
            lines.append(
                f"  note: block chosen with 0 incoming - attack scored {atk_score:.0f}"
            )
        elif incoming > 0 and kind == "attack" and blocks and needs_block_first:
            blk_score = max(blocks, key=lambda x: x[2])[2]
            lines.append(
                f"  note: attack over block-first - block scored {blk_score:.0f}"
            )
    return lines


def _finish_play(
    action: dict,
    reasons: list[str],
    playable: list[tuple[int, dict, float, list[str]]],
    chosen_index: int,
    incoming: int,
    needs_block_first: bool,
    path: str,
    card_reasons: list[str],
    kb,
) -> tuple[dict, list[str]]:
    comparison = _play_comparison_log(
        playable,
        chosen_index,
        incoming=incoming,
        needs_block_first=needs_block_first,
        path=path,
        kb=kb,
    )
    return action, reasons + comparison + card_reasons


def hand_select_session_key(state: dict) -> str:
    hs = state.get("hand_select") or {}
    mode = str(hs.get("mode") or "")
    prompt = str(hs.get("prompt") or "")[:96]
    cards = hs.get("cards") or (state.get("player") or {}).get("hand") or []
    sig = "|".join(
        str(c.get("id") or c.get("name") or i)
        for i, c in enumerate(cards[:15])
        if isinstance(c, dict)
    )
    return f"{mode}|{prompt}|{sig}"


def hand_select_selected_indices(hs: dict) -> set[int]:
    selected: set[int] = set()
    for entry in hs.get("selected_cards") or []:
        if isinstance(entry, dict) and entry.get("index") is not None:
            selected.add(int(entry["index"]))
        elif isinstance(entry, int):
            selected.add(int(entry))
    for raw in hs.get("selected_indices") or []:
        try:
            selected.add(int(raw))
        except (TypeError, ValueError):
            pass
    for list_idx, card in enumerate(hs.get("cards") or []):
        if not isinstance(card, dict):
            continue
        if card.get("is_selected") or card.get("selected"):
            idx = card.get("index")
            selected.add(int(idx) if idx is not None else list_idx)
    return selected


def clear_hand_select_session(state: dict) -> None:
    _hand_select_attempted.pop(hand_select_session_key(state), None)


def note_hand_select_action_failed(state: dict, action: dict) -> None:
    key = hand_select_session_key(state)
    name = str(action.get("action") or "")
    if name == "combat_select_card" and action.get("card_index") is not None:
        _hand_select_attempted.get(key, set()).discard(int(action["card_index"]))
    elif name == "combat_confirm_selection":
        _hand_select_attempted.pop(key, None)


def sync_hand_select_after_action(
    prev_state: dict, new_state: dict, action: dict
) -> None:
    if str(prev_state.get("state_type") or "").lower() != "hand_select":
        return
    key = hand_select_session_key(prev_state)
    name = str(action.get("action") or "")
    if str(new_state.get("state_type") or "").lower() != "hand_select":
        _hand_select_attempted.pop(key, None)
        return
    new_key = hand_select_session_key(new_state)
    if new_key != key:
        _hand_select_attempted.pop(key, None)
        key = new_key
    if name == "combat_select_card" and action.get("card_index") is not None:
        _hand_select_attempted.setdefault(key, set()).add(int(action["card_index"]))
    elif name == "combat_confirm_selection":
        new_hs = new_state.get("hand_select") or {}
        if not new_hs.get("can_confirm") and not hand_select_selected_indices(new_hs):
            _hand_select_attempted.pop(key, None)


def _decide_hand_select(state: dict) -> tuple[dict | None, list[str]]:
    hs = state.get("hand_select") or {}
    mode = str(hs.get("mode") or "").lower()
    prompt = str(hs.get("prompt") or "").lower()
    key = hand_select_session_key(state)
    selected_idx = hand_select_selected_indices(hs)
    attempted = _hand_select_attempted.get(key, set())
    reasons = [
        f"hand_select mode={mode} can_confirm={hs.get('can_confirm')} "
        f"selected={len(selected_idx)}"
    ]
    if selected_idx:
        reasons.append(f"selected_indices={sorted(selected_idx)}")
    if attempted:
        reasons.append(f"attempted={sorted(attempted)}")

    if hs.get("can_confirm"):
        return {"action": "combat_confirm_selection"}, reasons + ["confirm selection"]

    if selected_idx:
        return {"action": "combat_confirm_selection"}, reasons + [
            f"{len(selected_idx)} card(s) selected - confirm"
        ]

    cards = hs.get("cards") or (state.get("player") or {}).get("hand") or []
    kb = get_knowledge()
    is_upgrade = mode == "upgrade_select" or "upgrade" in prompt

    candidates: list[tuple[int, dict, float, list[str]]] = []
    for list_idx, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        if card.get("can_select") is False:
            continue
        if is_upgrade and card.get("is_upgraded"):
            continue
        api_index = int(card.get("index") if card.get("index") is not None else list_idx)
        if api_index in selected_idx or api_index in attempted:
            continue
        if is_upgrade:
            scored = smith_upgrade_priority(card, kb)
            score = scored.score
            pick_reasons = scored.reasons
        else:
            scored = score_combat_play(
                card,
                energy=3,
                incoming_damage=0,
                current_block=0,
                enemies=[],
                kb=kb,
                floor=int((state.get("run") or {}).get("floor") or 0),
            )
            score = scored.score
            pick_reasons = scored.reasons
        candidates.append((api_index, card, score, pick_reasons))

    if not candidates and attempted:
        _hand_select_attempted.pop(key, None)
        reasons.append("cleared attempted toggles - retry picks")
        for list_idx, card in enumerate(cards):
            if not isinstance(card, dict) or card.get("can_select") is False:
                continue
            if is_upgrade and card.get("is_upgraded"):
                continue
            api_index = int(card.get("index") if card.get("index") is not None else list_idx)
            if api_index in selected_idx:
                continue
            if is_upgrade:
                scored = smith_upgrade_priority(card, kb)
                score, pick_reasons = scored.score, scored.reasons
            else:
                scored = score_combat_play(
                    card,
                    energy=3,
                    incoming_damage=0,
                    current_block=0,
                    enemies=[],
                    kb=kb,
                    floor=int((state.get("run") or {}).get("floor") or 0),
                )
                score, pick_reasons = scored.score, scored.reasons
            candidates.append((api_index, card, score, pick_reasons))

    if not candidates:
        if hs.get("can_confirm") or selected_idx:
            return {"action": "combat_confirm_selection"}, reasons + [
                "no selectable cards - confirm"
            ]
        return None, reasons + ["no selectable cards - waiting"]

    candidates.sort(key=lambda x: x[2], reverse=True)
    api_index, best_card, _, pick_reasons = candidates[0]
    _hand_select_attempted.setdefault(key, set()).add(api_index)
    return (
        {"action": "combat_select_card", "card_index": api_index},
        reasons
        + [f"select {card_name(best_card)} (index {api_index})"]
        + pick_reasons,
    )


def _is_pure_block_play(card: dict, kb) -> bool:
    from sts2_agent.scorer import _hand_card_is_attack, _hand_card_is_block

    codex = kb.lookup_card(card_name(card))
    return _hand_card_is_block(card, codex) and not _hand_card_is_attack(card, codex)


def _is_block_play(card: dict, kb) -> bool:
    from sts2_agent.scorer import _hand_card_is_block

    return _hand_card_is_block(card, kb.lookup_card(card_name(card)))


def _is_debuff_play(card: dict, kb) -> bool:
    from sts2_agent.scorer import _hand_card_is_debuff

    return _hand_card_is_debuff(card, kb.lookup_card(card_name(card)))


def _is_attack_play(card: dict, kb) -> bool:
    from sts2_agent.scorer import _hand_card_is_attack

    return _hand_card_is_attack(card, kb.lookup_card(card_name(card)))


def _is_power_play(card: dict, kb) -> bool:
    from sts2_agent.scorer import _hand_card_is_power

    return _hand_card_is_power(card, kb.lookup_card(card_name(card)))


def _needs_target(card: dict, kb) -> bool:
    mode = str(card.get("target_type") or card.get("target") or "").lower().replace(" ", "")
    codex = kb.lookup_card(card_name(card))
    if codex:
        mode = str(codex.get("target") or mode).lower().replace(" ", "")
    if mode in ("none", "self", "all_enemies", "allenemies", "everyenemy"):
        return False
    if "all" in mode and "enemy" in mode:
        return False
    if mode in ("enemy", "anyenemy", "single_enemy"):
        return True
    return _is_attack_play(card, kb) and "all" not in mode


def _build_play_action(
    card_index: int,
    card: dict,
    enemies: list[dict],
    kb,
    *,
    prefer_vulnerable: bool = False,
    prefer_entity_id: str | None = None,
) -> dict | None:
    action: dict[str, Any] = {"action": "play_card", "card_index": card_index}
    if not _needs_target(card, kb):
        return action

    living = [e for e in enemies if int(e.get("hp") or 0) > 0]
    if prefer_entity_id:
        match = [e for e in living if e.get("entity_id") == prefer_entity_id]
        if match:
            action["target"] = prefer_entity_id
            return action

    if prefer_vulnerable:
        vuln = [e for e in living if enemy_has_vulnerable(e)]
        if vuln:
            living = vuln

    target, _ = pick_highest_threat_enemy(living)
    if not target or not target.get("entity_id"):
        return None
    action["target"] = target["entity_id"]
    return action


def decide_combat_potion_from_context(
    state: dict,
    *,
    player: dict,
    living: list[dict],
    incoming: int,
    block: int,
    lethal_target: dict | None,
    lethal_damage: int,
    has_playable_cards: bool,
) -> tuple[dict | None, list[str]]:
    from sts2_agent.potions import CombatPotionContext, evaluate_combat_potions, player_hp_ratio

    hp, max_hp, hp_ratio = player_hp_ratio(player)
    gap = max(incoming - block, 0)
    ctx = CombatPotionContext(
        player=player,
        enemies=living,
        incoming=incoming,
        block=block,
        lethal_target=lethal_target,
        lethal_damage=lethal_damage,
        has_playable_cards=has_playable_cards,
        hp=hp,
        max_hp=max_hp,
        hp_ratio=hp_ratio,
        gap=gap,
    )
    return evaluate_combat_potions(ctx)


def decide_combat_potion(state: dict) -> tuple[dict | None, list[str]]:
    """Potion use for combat - also called before BC policy so emergency heals fire."""
    battle = state.get("battle") or {}
    if battle.get("is_play_phase") is False:
        return None, ["potions: skipped - not play phase"]

    kb = get_knowledge()
    player = state.get("player") or {}
    hand = player.get("hand") or []
    energy = int(player.get("energy") or player.get("current_energy") or 0)
    block = int(player.get("block") or 0)
    living = living_enemies(battle)
    if not living:
        return None, ["potions: skipped - no living enemies"]

    incoming, _ = total_incoming_attack_damage(living)
    floor = int((state.get("run") or {}).get("floor") or 0)

    lethal_damage, _ = estimate_hand_damage(hand, energy, living, kb)
    lethal_target, _ = find_lethal_target(living, lethal_damage)

    playable = False
    for card in hand:
        if score_combat_play(
            card,
            energy=energy,
            incoming_damage=incoming,
            current_block=block,
            enemies=living,
            kb=kb,
            floor=floor,
        ).score > 0:
            playable = True
            break

    return decide_combat_potion_from_context(
        state,
        player=player,
        living=living,
        incoming=incoming,
        block=block,
        lethal_target=lethal_target,
        lethal_damage=lethal_damage,
        has_playable_cards=playable,
    )
