"""Combat turn solver: exact play-sequence search + next-turn pile lookahead."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sts2_agent.enemy_patterns import enrich_incoming_damage
from sts2_agent.knowledge import KnowledgeBase, get_knowledge
from sts2_agent.pile_odds import next_turn_combat_estimates
from sts2_agent.scorer import (
    _card_block_value,
    _card_cost,
    _card_damage_value,
    _hand_card_is_attack,
    _hand_card_is_block,
    card_applies_vulnerable,
    card_name,
    enemy_has_vulnerable,
    enemy_incoming_attack_damage,
    total_incoming_attack_damage,
)
from sts2_agent.state_parse import living_enemies

logger = logging.getLogger(__name__)

MAX_PLAY_DEPTH = 8
SCALING_MOVE_TOKENS = (
    "spawn",
    "summon",
    "strength",
    "scaling",
    "divide",
    "split",
    "grow",
    "enrage",
    "empower",
    "ritual",
    "revive",
)


@dataclass
class PlannedPlay:
    """Stable identity for a play step (hand indices shift after each play)."""

    card_key: str
    card_label: str
    target_entity_id: str | None = None


@dataclass
class TurnPlan:
    steps: list[PlannedPlay]
    tag: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class SequenceOutcome:
    steps: list[PlannedPlay]
    total_damage: int = 0
    total_block: int = 0
    hp_lost: int = 0
    energy_left: int = 0
    cards_played: int = 0
    kills: list[str] = field(default_factory=list)
    enemy_hp_after: dict[str, int] = field(default_factory=dict)

    @property
    def is_lethal(self) -> bool:
        """True when every enemy is dead after this sequence."""
        if not self.enemy_hp_after:
            return False
        return all(hp <= 0 for hp in self.enemy_hp_after.values())

    @property
    def has_kills(self) -> bool:
        return bool(self.kills)


@dataclass
class SolverContext:
    state: dict
    kb: KnowledgeBase
    hand: list[dict]
    energy: int
    player_block: int
    player_hp: int
    incoming: int
    next_incoming: int
    living: list[dict]
    next_expected_damage: int = 0
    next_expected_block: int = 0
    enemy_scales: bool = False
    scale_reasons: list[str] = field(default_factory=list)


_turn_plans: dict[str, TurnPlan] = {}
_solver_enabled = True


def set_combat_solver_enabled(enabled: bool) -> None:
    global _solver_enabled
    _solver_enabled = bool(enabled)


def combat_solver_enabled() -> bool:
    return _solver_enabled


def clear_turn_plan_cache() -> None:
    _turn_plans.clear()


def _card_key(card: dict) -> str:
    return str(card.get("id") or card.get("name") or card_name(card)).upper()


def _turn_cache_key(state: dict) -> str:
    battle = state.get("battle") or {}
    run = state.get("run") or {}
    return "|".join(
        [
            str(run.get("id") or run.get("run_id") or ""),
            str(battle.get("round") or ""),
            str(battle.get("turn") or ""),
        ]
    )


def _player_status_strength(player: dict) -> int:
    total = 0
    for status in player.get("status") or player.get("powers") or []:
        if isinstance(status, dict):
            name = str(status.get("id") or status.get("name") or status.get("power") or "")
            stacks = status.get("amount", status.get("stacks", status.get("count", 1)))
        else:
            name = str(status)
            stacks = 1
        if "strength" in name.lower():
            try:
                total += int(stacks)
            except (TypeError, ValueError):
                total += 1
    return max(total, 0)


def _player_outgoing_multiplier(player: dict) -> float:
    mult = 1.0
    for status in player.get("status") or player.get("powers") or []:
        if isinstance(status, dict):
            name = str(status.get("id") or status.get("name") or status.get("power") or "")
        else:
            name = str(status)
        if "weak" in name.lower():
            mult *= 0.75
    return mult


def _hits_all_enemies(card: dict, kb: KnowledgeBase) -> bool:
    mode = str(card.get("target_type") or card.get("target") or "").lower().replace(" ", "")
    codex = kb.lookup_card(card_name(card))
    if codex:
        mode = str(codex.get("target") or mode).lower().replace(" ", "")
    if mode in ("all_enemies", "allenemies", "everyenemy"):
        return True
    return "all" in mode and "enemy" in mode


def _needs_target(card: dict, kb: KnowledgeBase) -> bool:
    from sts2_agent.combat import _is_attack_play, _needs_target

    return _needs_target(card, kb)


def _is_playable(card: dict, energy: int) -> bool:
    if card.get("can_play") is False or card.get("playable") is False:
        return False
    cost = _card_cost(card)
    if cost > energy:
        return False
    if cost >= 99:
        return False
    return True


def _enemy_ids(living: list[dict]) -> list[str]:
    out: list[str] = []
    for enemy in living:
        eid = enemy.get("entity_id") or enemy.get("id")
        if eid and int(enemy.get("hp") or 0) > 0:
            out.append(str(eid))
    return out


@dataclass
class _SimState:
    energy: int
    enemy_hp: dict[str, int]
    enemy_vuln: set[str]
    block_gained: int
    strength: int
    outgoing_mult: float
    used: frozenset[int]
    steps: list[PlannedPlay]
    total_damage: int = 0


def _simulate_play(
    sim: _SimState,
    hand: list[dict],
    kb: KnowledgeBase,
    card_index: int,
    target_id: str | None,
) -> _SimState | None:
    card = hand[card_index]
    if not _is_playable(card, sim.energy):
        return None

    codex = kb.lookup_card(card_name(card))
    cost = _card_cost(card)
    enemy_hp = dict(sim.enemy_hp)
    enemy_vuln = set(sim.enemy_vuln)
    block_gained = sim.block_gained
    strength = sim.strength
    total_damage = sim.total_damage

    if card_applies_vulnerable(card, kb):
        if target_id:
            enemy_vuln.add(target_id)
        elif _hits_all_enemies(card, kb):
            enemy_vuln.update(eid for eid, hp in enemy_hp.items() if hp > 0)

    if _hand_card_is_attack(card, codex):
        base = _card_damage_value(card, codex)
        hit_damage = int(base * sim.outgoing_mult) + strength
        if _hits_all_enemies(card, kb):
            for eid, hp in list(enemy_hp.items()):
                if hp <= 0:
                    continue
                dmg = hit_damage
                if eid in enemy_vuln or enemy_has_vulnerable({"entity_id": eid, "status": []}):
                    dmg = int(dmg * 1.5)
                enemy_hp[eid] = max(0, hp - dmg)
                total_damage += dmg
        elif target_id and target_id in enemy_hp:
            dmg = hit_damage
            if target_id in enemy_vuln:
                dmg = int(dmg * 1.5)
            enemy_hp[target_id] = max(0, enemy_hp[target_id] - dmg)
            total_damage += dmg
        else:
            return None

    if _hand_card_is_block(card, codex):
        block_gained += _card_block_value(card, codex)

    # Temporary strength from powers (rough: codex strength powers).
    if codex:
        for power in codex.get("powers_applied") or []:
            if isinstance(power, dict):
                pname = str(power.get("power") or "").lower()
                amt = int(power.get("amount") or power.get("stacks") or 1)
                if "strength" in pname:
                    strength += amt

    step = PlannedPlay(
        card_key=_card_key(card),
        card_label=card_name(card),
        target_entity_id=target_id,
    )
    return _SimState(
        energy=sim.energy - cost,
        enemy_hp=enemy_hp,
        enemy_vuln=enemy_vuln,
        block_gained=block_gained,
        strength=strength,
        outgoing_mult=sim.outgoing_mult,
        used=sim.used | {card_index},
        steps=sim.steps + [step],
        total_damage=total_damage,
    )


def _incoming_from_survivors(
    enemy_hp: dict[str, int],
    living: list[dict],
) -> int:
    """Sum attack damage only from enemies still alive after the play sequence."""
    total = 0
    for enemy in living:
        eid = enemy.get("entity_id") or enemy.get("id")
        if not eid:
            continue
        key = str(eid)
        if enemy_hp.get(key, int(enemy.get("hp") or 0)) <= 0:
            continue
        total += enemy_incoming_attack_damage(enemy)
    return total


def _finalize_outcome(sim: _SimState, ctx: SolverContext) -> SequenceOutcome:
    incoming_after = _incoming_from_survivors(sim.enemy_hp, ctx.living)
    effective_block = ctx.player_block + sim.block_gained
    hp_lost = max(0, incoming_after - effective_block)
    kills = [eid for eid, hp in sim.enemy_hp.items() if hp <= 0]
    return SequenceOutcome(
        steps=list(sim.steps),
        total_damage=sim.total_damage,
        total_block=sim.block_gained,
        hp_lost=hp_lost,
        energy_left=sim.energy,
        cards_played=len(sim.steps),
        kills=kills,
        enemy_hp_after=dict(sim.enemy_hp),
    )


def _initial_enemy_hp(living: list[dict]) -> dict[str, int]:
    hp_map: dict[str, int] = {}
    for enemy in living:
        eid = enemy.get("entity_id") or enemy.get("id")
        if not eid:
            continue
        hp = int(enemy.get("hp") or 0)
        if hp > 0:
            hp_map[str(eid)] = hp
    return hp_map


def _enumerate_sequences(ctx: SolverContext) -> list[SequenceOutcome]:
    hand = ctx.hand
    if not hand:
        return [
            SequenceOutcome(
                steps=[],
                hp_lost=max(0, ctx.incoming - ctx.player_block),
                energy_left=ctx.energy,
                enemy_hp_after=_initial_enemy_hp(ctx.living),
            )
        ]

    player = ctx.state.get("player") or {}
    initial = _SimState(
        energy=ctx.energy,
        enemy_hp=_initial_enemy_hp(ctx.living),
        enemy_vuln={
            str(e.get("entity_id") or e.get("id"))
            for e in ctx.living
            if enemy_has_vulnerable(e) and (e.get("entity_id") or e.get("id"))
        },
        block_gained=0,
        strength=_player_status_strength(player),
        outgoing_mult=_player_outgoing_multiplier(player),
        used=frozenset(),
        steps=[],
    )

    outcomes: list[SequenceOutcome] = []
    seen: set[tuple] = set()

    def dfs(sim: _SimState) -> None:
        sig = (sim.used, sim.energy, tuple(sorted(sim.enemy_hp.items())))
        if sig in seen:
            return
        seen.add(sig)
        outcomes.append(_finalize_outcome(sim, ctx))

        if len(sim.steps) >= MAX_PLAY_DEPTH:
            return

        for index, card in enumerate(hand):
            if index in sim.used:
                continue
            if not _is_playable(card, sim.energy):
                continue

            alive_ids = [eid for eid, hp in sim.enemy_hp.items() if hp > 0]
            targets: list[str | None]
            if _hand_card_is_attack(card, ctx.kb.lookup_card(card_name(card))):
                if _hits_all_enemies(card, ctx.kb):
                    targets = [None]
                elif alive_ids:
                    targets = list(alive_ids)
                else:
                    continue
            elif _needs_target(card, ctx.kb):
                targets = list(alive_ids) or [None]
            else:
                targets = [None]

            for target_id in targets:
                nxt = _simulate_play(sim, hand, ctx.kb, index, target_id)
                if nxt is not None:
                    dfs(nxt)

    dfs(initial)
    if not outcomes:
        outcomes.append(_finalize_outcome(initial, ctx))
    return outcomes


def _compendium_enemy_scales(living: list[dict]) -> tuple[bool, list[str]]:
    from sts2_agent.enemy_compendium import get_compendium_kb

    reasons: list[str] = []
    kb = get_compendium_kb()
    for enemy in living:
        entry = kb.lookup_enemy(enemy)
        if not entry:
            continue
        for move_key, move in (entry.get("moves") or {}).items():
            blob = f"{move_key}|{move.get('api_label') or ''}".lower()
            tags = [str(t).lower() for t in (move.get("tags") or [])]
            if any(tok in blob for tok in SCALING_MOVE_TOKENS):
                reasons.append(f"scaling move {move_key} on {enemy.get('name')}")
                return True, reasons
            if any("buff:strength" in t or "spawn" in t for t in tags):
                reasons.append(f"scaling tags on {enemy.get('name')}: {tags}")
                return True, reasons
    return False, reasons


def _qwen_scaling_override() -> tuple[bool, list[str]]:
    try:
        from sts2_agent.qwen_advisor import get_qwen_advisor, is_qwen_combat_enabled

        if not is_qwen_combat_enabled():
            return False, []
        mult = get_qwen_advisor().get_multipliers()
        if mult.strategy == "aggressive":
            return True, ["Qwen strategy override: aggressive"]
    except Exception:
        pass
    return False, []


def _lethal_next_turn(
    outcome: SequenceOutcome,
    ctx: SolverContext,
) -> bool:
    if not ctx.living:
        return False
    remaining_hps = [hp for hp in outcome.enemy_hp_after.values() if hp > 0]
    if not remaining_hps:
        return False
    min_hp = min(remaining_hps)
    return ctx.next_expected_damage >= min_hp


def _hp_cost_survivable(hp_lost: int, player_hp: int) -> bool:
    if player_hp <= 0:
        return False
    if hp_lost <= 0:
        return True
    return hp_lost < player_hp and (player_hp - hp_lost) > max(8, player_hp // 4)


def _trade_score(outcome: SequenceOutcome, ctx: SolverContext) -> float:
    incoming_after = _incoming_from_survivors(outcome.enemy_hp_after, ctx.living)
    unblocked_threat = max(0, incoming_after - ctx.player_block)
    useful_block = min(outcome.total_block, unblocked_threat)
    return (
        outcome.total_damage * 3.0
        + useful_block * 2.0
        - outcome.hp_lost * 4.0
        - ctx.next_incoming * 0.5
    )


def _pick_best_sequence(
    outcomes: list[SequenceOutcome],
    ctx: SolverContext,
) -> tuple[SequenceOutcome, str, list[str]]:
    reasons: list[str] = []
    lethal = [o for o in outcomes if o.is_lethal]
    if lethal:
        best = min(
            lethal,
            key=lambda o: (
                o.hp_lost,
                len(o.steps),
                -o.total_damage,
            ),
        )
        reasons.append(
            f"lethal T1: {len(best.steps)} plays, dmg={best.total_damage}, hp_lost={best.hp_lost}"
        )
        return best, "solver: lethal T1", reasons

    # Killing attackers this turn voids their damage — prefer that over pointless block.
    kill_setup = [
        o
        for o in outcomes
        if o.has_kills and o.hp_lost == 0 and _hp_cost_survivable(o.hp_lost, ctx.player_hp)
    ]
    if kill_setup:
        best = max(
            kill_setup,
            key=lambda o: (len(o.kills), o.total_damage, -len(o.steps)),
        )
        reasons.append(
            f"kill removes incoming: {len(best.kills)} kill(s), dmg={best.total_damage}, "
            f"hp_lost={best.hp_lost}"
        )
        return best, "solver: kill removes incoming", reasons

    setup_candidates = [
        o
        for o in outcomes
        if _lethal_next_turn(o, ctx) and _hp_cost_survivable(o.hp_lost, ctx.player_hp)
    ]
    if setup_candidates:
        best = max(
            setup_candidates,
            key=lambda o: (o.total_damage, -o.hp_lost, o.total_block),
        )
        reasons.append(
            f"setup T2 lethal: dmg={best.total_damage}, hp_lost={best.hp_lost}, "
            f"next_est_dmg={ctx.next_expected_damage}"
        )
        return best, "solver: setup T2 lethal", reasons

    if ctx.enemy_scales:
        best = max(
            outcomes,
            key=lambda o: (o.total_damage, -o.hp_lost),
        )
        reasons.extend(ctx.scale_reasons)
        reasons.append(
            f"aggressive: dmg={best.total_damage}, hp_lost={best.hp_lost}"
        )
        return best, "solver: aggressive (scales)", reasons

    best = max(
        outcomes,
        key=lambda o: (_trade_score(o, ctx), -len(o.steps), o.total_damage),
    )
    reasons.append(
        f"trade: dmg={best.total_damage}, block={best.total_block}, hp_lost={best.hp_lost}"
    )
    return best, "solver: trade", reasons


def _build_context(state: dict) -> SolverContext | None:
    battle = state.get("battle") or {}
    player = state.get("player") or {}
    living = living_enemies(battle)
    hand = list(player.get("hand") or [])
    energy = int(player.get("energy") or player.get("current_energy") or 0)
    player_block = int(player.get("block") or 0)
    player_hp = int(player.get("hp") or 0)

    incoming, _ = total_incoming_attack_damage(living)
    next_incoming = 0
    try:
        from sts2_agent.combat import compendium_decisions_enabled

        if compendium_decisions_enabled():
            _kb_this, next_incoming, _ = enrich_incoming_damage(living, state)
    except Exception:
        pass

    kb = get_knowledge()
    next_est = next_turn_combat_estimates(player, kb, energy=energy)

    scales, scale_reasons = _compendium_enemy_scales(living)
    qwen_scale, qwen_reasons = _qwen_scaling_override()
    if qwen_scale:
        scales = True
        scale_reasons.extend(qwen_reasons)

    return SolverContext(
        state=state,
        kb=kb,
        hand=hand,
        energy=energy,
        player_block=player_block,
        player_hp=player_hp,
        incoming=incoming,
        next_incoming=next_incoming,
        living=living,
        next_expected_damage=next_est.expected_damage,
        next_expected_block=next_est.expected_block,
        enemy_scales=scales,
        scale_reasons=scale_reasons,
    )


def _planned_to_action(
    step: PlannedPlay,
    hand: list[dict],
    living: list[dict],
    kb: KnowledgeBase,
) -> dict | None:
    from sts2_agent.combat import _build_play_action

    for index, card in enumerate(hand):
        if _card_key(card) != step.card_key and card_name(card) != step.card_label:
            continue
        return _build_play_action(
            index,
            card,
            living,
            kb,
            prefer_entity_id=step.target_entity_id,
        )
    return None


def _solve_turn(ctx: SolverContext) -> TurnPlan:
    outcomes = _enumerate_sequences(ctx)
    best, tag, pick_reasons = _pick_best_sequence(outcomes, ctx)

    reasons = [
        f"enumerated {len(outcomes)} sequences (depth≤{MAX_PLAY_DEPTH})",
        *pick_reasons,
    ]
    reasons.extend(ctx.scale_reasons)

    if not best.steps:
        return TurnPlan(steps=[], tag=tag, reasons=reasons)

    return TurnPlan(steps=list(best.steps), tag=tag, reasons=reasons)


def try_solver_decide(state: dict) -> tuple[dict | None, list[str]] | None:
    """Return (action, reasons) when the solver handles this poll; None → legacy path."""
    if not _solver_enabled:
        return None

    cache_key = _turn_cache_key(state)
    cached = _turn_plans.get(cache_key)
    if cached and cached.steps:
        step = cached.steps.pop(0)
        player = state.get("player") or {}
        hand = player.get("hand") or []
        living = living_enemies(state.get("battle") or {})
        kb = get_knowledge()
        action = _planned_to_action(step, hand, living, kb)
        reasons = [cached.tag, "solver: executing cached plan", f"play {step.card_label}"]
        if not cached.steps:
            _turn_plans.pop(cache_key, None)
        if action:
            return action, reasons
        logger.warning("solver cache step not found in hand: %s", step.card_label)
        _turn_plans.pop(cache_key, None)

    ctx = _build_context(state)
    if ctx is None:
        return None

    if not ctx.hand and not ctx.living:
        return {"action": "end_turn"}, ["solver: no hand, no enemies — end turn"]

    if not ctx.hand:
        return {"action": "end_turn"}, ["solver: empty hand — end turn"]

    try:
        plan = _solve_turn(ctx)
    except Exception as exc:
        logger.warning("combat solver solve failed: %s", exc)
        return None

    logger.info(plan.tag)

    if not plan.steps:
        return {"action": "end_turn"}, plan.reasons + [plan.tag, "solver: no plays — end turn"]

    cache_key = _turn_cache_key(state)
    if len(plan.steps) > 1:
        _turn_plans[cache_key] = TurnPlan(
            steps=plan.steps[1:],
            tag=plan.tag,
            reasons=plan.reasons,
        )

    step = plan.steps[0]
    action = _planned_to_action(step, ctx.hand, ctx.living, ctx.kb)
    if not action:
        return None

    return action, [plan.tag, *plan.reasons, f"play {step.card_label}"]
