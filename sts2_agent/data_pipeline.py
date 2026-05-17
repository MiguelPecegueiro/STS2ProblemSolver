"""Append-only JSONL run/decision logging."""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sts2_agent.characters import normalize_character_name

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DECISIONS_PATH = DATA_DIR / "decisions.jsonl"
RUNS_PATH = DATA_DIR / "runs.jsonl"
MAX_BUFFER = 1000

_collector_instance: str | None = None


def configure_data_paths(
    data_dir: str | Path | None = None,
    instance_id: str | int | None = None,
) -> Path:
    """Route JSONL output to a per-instance directory (parallel collection)."""
    global DATA_DIR, DECISIONS_PATH, RUNS_PATH, _collector_instance, _pipeline

    if data_dir is None and instance_id is not None:
        data_dir = PROJECT_ROOT / "data" / "instances" / str(instance_id)
    elif data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    else:
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = PROJECT_ROOT / data_dir

    DATA_DIR = data_dir
    DECISIONS_PATH = DATA_DIR / "decisions.jsonl"
    RUNS_PATH = DATA_DIR / "runs.jsonl"
    _collector_instance = str(instance_id) if instance_id is not None else None

    if _pipeline is not None:
        try:
            _pipeline.flush_on_exit()
        except Exception:
            pass
    _pipeline = None

    if instance_id is not None:
        from sts2_agent.enemy_compendium import set_compendium_writes_enabled

        set_compendium_writes_enabled(False)

    return DATA_DIR


def get_collector_instance() -> str | None:
    return _collector_instance

COMBAT_TYPES = frozenset({"monster", "elite", "boss", "hand_select"})
ENEMY_INTENT_HISTORY_LEN = 3

# Default tag for runs before main.py calls set_agent_version() (keep in sync with main.AGENT_VERSION_RULES).
DEFAULT_AGENT_VERSION = "rules_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _card_id(card: dict) -> str:
    return str(card.get("id") or card.get("name") or "UNKNOWN").upper().replace(" ", "_")


def extract_deck_card_ids(state: dict) -> list[str]:
    """Collect deck card IDs from player.deck or all visible piles."""
    player = state.get("player") or {}
    deck = player.get("deck")
    if isinstance(deck, list) and deck:
        return [
            _card_id(card) if isinstance(card, dict) else str(card).upper()
            for card in deck
        ]

    seen: set[str] = set()
    ordered: list[str] = []
    for pile_key in ("draw_pile", "discard_pile", "hand", "exhaust_pile"):
        for card in player.get(pile_key) or []:
            if not isinstance(card, dict):
                continue
            cid = _card_id(card)
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
    return ordered


def extract_potion_ids(state: dict) -> list[str]:
    """Potion IDs/names from player belt."""
    player = state.get("player") or {}
    potions: list[str] = []
    for potion in player.get("potions") or []:
        if not potion or potion is False:
            continue
        if isinstance(potion, dict):
            pid = str(potion.get("id") or potion.get("name") or "")
        else:
            pid = str(potion)
        if pid:
            potions.append(pid)
    return potions


def extract_relic_ids(state: dict) -> list[str]:
    player = state.get("player") or {}
    relics: list[str] = []
    for relic in player.get("relics") or []:
        if isinstance(relic, dict):
            rid = str(relic.get("id") or relic.get("name") or "")
        else:
            rid = str(relic)
        if rid:
            relics.append(rid)
    return relics


def build_state_snapshot(
    state: dict,
    *,
    enemy_intent_histories: dict[str, list[dict]] | None = None,
) -> dict:
    """Compact state snapshot for RL training."""
    from sts2_agent.enemy_compendium import compact_enemy_intent

    player = state.get("player") or {}
    battle = state.get("battle") or {}
    state_type = str(state.get("state_type") or "").lower()
    in_combat = state_type in COMBAT_TYPES
    histories = enemy_intent_histories or {}

    hand = []
    for card in player.get("hand") or []:
        if not isinstance(card, dict):
            continue
        hand.append(
            {
                "id": _card_id(card),
                "name": card.get("name"),
                "cost": card.get("cost"),
                "type": card.get("type"),
            }
        )

    living_peers = [
        e
        for e in battle.get("enemies") or []
        if isinstance(e, dict) and int(e.get("hp") or 0) > 0
    ]
    enemies = []
    for enemy in battle.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        compact = compact_enemy_intent(enemy, peers=living_peers)
        intent_type = ""
        intent_value = None
        if compact:
            intent_type = str(compact.get("intent") or "")
            intent_value = compact.get("damage")
            if compact.get("block"):
                intent_value = intent_value or compact.get("block")
        else:
            intents = enemy.get("intents") or []
            if isinstance(enemy.get("intent"), dict):
                intents = [enemy["intent"]]
            if intents and isinstance(intents[0], dict):
                intent_type = str(intents[0].get("type") or intents[0].get("title") or "")
                intent_value = intents[0].get("damage") or intents[0].get("label")
        entity_id = str(enemy.get("entity_id") or enemy.get("id") or "")
        history = [
            {**entry, "entity_id": entry.get("entity_id") or entity_id}
            for entry in list(histories.get(entity_id, []))[-ENEMY_INTENT_HISTORY_LEN:]
        ]
        enemies.append(
            {
                "id": enemy.get("entity_id"),
                "entity_id": entity_id or None,
                "name": enemy.get("name"),
                "compendium_key": (compact or {}).get("compendium_key"),
                "role": (compact or {}).get("role"),
                "hp": enemy.get("hp"),
                "max_hp": enemy.get("max_hp"),
                "block": enemy.get("block"),
                "intent": intent_type,
                "intent_value": intent_value,
                "intent_tags": (compact or {}).get("tags") or [],
                "intent_history": history,
            }
        )

    relics = []
    for relic in player.get("relics") or []:
        if isinstance(relic, dict):
            relics.append(str(relic.get("id") or relic.get("name") or ""))
        else:
            relics.append(str(relic))

    potions = []
    for potion in player.get("potions") or []:
        if potion and potion is not False:
            if isinstance(potion, dict):
                potions.append(str(potion.get("id") or potion.get("name") or ""))
            else:
                potions.append(str(potion))

    status_effects = []
    for status in player.get("status") or player.get("powers") or []:
        if isinstance(status, dict):
            status_effects.append(str(status.get("id") or status.get("name") or status))
        else:
            status_effects.append(str(status))

    draw_pile_count = (
        player.get("draw_pile_count")
        if player.get("draw_pile_count") is not None
        else len(player.get("draw_pile") or [])
    )
    discard_pile_count = (
        player.get("discard_pile_count")
        if player.get("discard_pile_count") is not None
        else len(player.get("discard_pile") or [])
    )

    snapshot: dict[str, Any] = {
        "player_hp": player.get("hp"),
        "player_max_hp": player.get("max_hp"),
        "player_block": _safe_int(player.get("block"), 0) if in_combat else 0,
        "player_energy": _safe_int(player.get("energy"), 0) if in_combat else 0,
        "hand": hand,
        "draw_pile_count": draw_pile_count,
        "discard_pile_count": discard_pile_count,
        "relics": [r for r in relics if r],
        "potions": [p for p in potions if p],
        "status_effects": status_effects,
        "enemies": enemies,
        "attack_ratio_in_draw": 0.0,
        "block_ratio_in_draw": 0.0,
        "high_value_cards_in_draw": 0.0,
        "expected_block_next_turn": 0,
        "expected_damage_next_turn": 0,
    }

    if in_combat:
        from sts2_agent.knowledge import get_knowledge
        from sts2_agent.pile_odds import draw_pile_feature_summary

        pile_feats = draw_pile_feature_summary(
            player,
            get_knowledge(),
            energy=_safe_int(player.get("energy"), 0),
        )
        snapshot.update(pile_feats)

    return snapshot


class DataPipeline:
    """Collects state-action pairs and run summaries for RL training."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self._buffer: deque[dict] = deque(maxlen=MAX_BUFFER)
        self._write_queue: queue.Queue[tuple[Path, str] | None] = queue.Queue()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

        # Run tracking
        self._run_active = False
        self._character: str = "Unknown"
        self._ascension: int = 0
        self._decision_count = 0
        self._damage_taken_run = 0
        self._damage_dealt_run = 0
        self._last_hp: int | None = None
        self._max_floor_seen = 0
        self._max_act_seen = 1
        self._last_state_type: str | None = None
        self._last_run_summary: dict | None = None
        self._agent_version: str = DEFAULT_AGENT_VERSION
        self._run_started_at: datetime | None = None

        # Combat session tracking
        self._in_combat = False
        self._combat_start_hp: int | None = None
        self._combat_start_max_hp: int | None = None
        self._combat_enemy_ids: set[str] = set()
        self._combat_state_type: str = ""
        self._combat_decision_indices: list[int] = []
        self._enemy_hp_snapshot: dict[str, int] = {}
        self._last_deck: list[str] = []
        self._last_relics: list[str] = []

        # HP conservation tracking
        self._hp_before_each_combat: list[int] = []
        self._hp_after_each_combat: list[int] = []
        self._max_hp_each_combat: list[int] = []
        self._combat_rewards: list[float] = []
        self._bosses_killed = 0
        self._combat_prev_hp: int | None = None
        self._combat_prev_block: int | None = None
        self._combat_damage_since_decision = 0
        self._enemy_intent_history: dict[str, list[dict]] = {}

        DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    def _writer_loop(self) -> None:
        while True:
            job = self._write_queue.get()
            if job is None:
                break
            path, line = job
            try:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as exc:
                logger.debug("data_pipeline write failed: %s", exc)
            finally:
                self._write_queue.task_done()

    def _enqueue_line(self, path: Path, obj: dict) -> None:
        try:
            line = json.dumps(obj, ensure_ascii=False, default=str) + "\n"
            self._write_queue.put((path, line))
        except Exception as exc:
            logger.debug("data_pipeline serialize failed: %s", exc)

    def observe_state(self, state: dict) -> None:
        """Detect run boundaries and combat transitions."""
        try:
            self._observe_state_inner(state)
        except Exception as exc:
            logger.debug("data_pipeline observe_state failed: %s", exc)

    def _observe_state_inner(self, state: dict) -> None:
        state_type = str(state.get("state_type") or "").lower()
        run = state.get("run") or {}
        player = state.get("player") or {}
        floor = _safe_int(run.get("floor"))
        act = _safe_int(run.get("act"), 1)

        if floor > self._max_floor_seen:
            self._max_floor_seen = floor
        if act > self._max_act_seen:
            self._max_act_seen = act

        # Start run: first in-run screen after menu / new floor 1
        if not self._run_active and self._should_start_run(state_type, floor, run):
            self.start_run(state)

        # Expose pipeline run_id on state so combat / pattern verification can use it.
        if self._run_active and self.run_id:
            run_obj = state.get("run")
            if not isinstance(run_obj, dict):
                run_obj = {}
                state["run"] = run_obj
            run_obj["run_id"] = self.run_id

        # Track HP deltas for run stats
        hp = _safe_int(player.get("hp"))
        if self._run_active and self._last_hp is not None and hp < self._last_hp:
            self._damage_taken_run += self._last_hp - hp
        self._last_hp = hp if player else self._last_hp

        # Remember latest deck / relics for run summary
        if self._run_active and player:
            deck = extract_deck_card_ids(state)
            if deck:
                self._last_deck = deck
            relics = extract_relic_ids(state)
            if relics:
                self._last_relics = relics

        # Combat lifecycle
        in_combat_now = state_type in COMBAT_TYPES
        if in_combat_now and not self._in_combat:
            self._begin_combat(state)
        elif self._in_combat and in_combat_now:
            self._track_enemy_damage(state)
            self._track_enemy_intent_history(state)
        elif self._in_combat and not in_combat_now:
            self._track_enemy_damage(state)
            self._end_combat(state)

        # End run
        if self._run_active and self._should_end_run(state_type, player):
            won = state_type != "game_over" and hp > 0
            cause = self._infer_death_cause(state_type, state)
            self.end_run(state, won=won, cause_of_death=cause)

        self._last_state_type = state_type

    def _should_start_run(self, state_type: str, floor: int, run: dict) -> bool:
        if state_type in ("menu", "game_over", ""):
            return False
        if floor >= 1 or run:
            return True
        return state_type in ("map", "monster", "elite", "boss", "event", "shop", "rest_site")

    def _should_end_run(self, state_type: str, player: dict) -> bool:
        if state_type == "game_over":
            return True
        if state_type == "menu" and self._max_floor_seen > 0:
            return True
        if state_type in COMBAT_TYPES and _safe_int(player.get("hp")) <= 0:
            return True
        return False

    def start_run(self, state: dict | None = None) -> str:
        self.run_id = str(uuid4())
        self._run_active = True
        self._buffer.clear()
        self._decision_count = 0
        self._damage_taken_run = 0
        self._damage_dealt_run = 0
        self._max_floor_seen = 0
        self._max_act_seen = 1
        self._in_combat = False
        self._combat_decision_indices.clear()
        self._enemy_hp_snapshot = {}
        self._last_deck = []
        self._last_relics = []
        self._hp_before_each_combat = []
        self._hp_after_each_combat = []
        self._max_hp_each_combat = []
        self._combat_rewards = []
        self._bosses_killed = 0
        self._combat_prev_hp = None
        self._combat_prev_block = None
        self._combat_damage_since_decision = 0
        self._run_started_at = datetime.now(timezone.utc)

        if state:
            player = state.get("player") or {}
            run = state.get("run") or {}
            self._character = normalize_character_name(player.get("character"))
            self._ascension = _safe_int(run.get("ascension"))
            self._last_hp = _safe_int(player.get("hp"))
            self._max_floor_seen = _safe_int(run.get("floor"))
            self._max_act_seen = _safe_int(run.get("act"), 1)
            deck = extract_deck_card_ids(state)
            if deck:
                self._last_deck = deck
            relics = extract_relic_ids(state)
            if relics:
                self._last_relics = relics

        logger.info("data_pipeline: started run %s", self.run_id)
        return self.run_id

    def end_run(
        self,
        state: dict | None,
        *,
        won: bool,
        cause_of_death: str | None = None,
    ) -> None:
        if not self.run_id or not self._run_active:
            return

        if self._in_combat:
            self._end_combat(state or {})

        outcome = "won" if won else "lost"
        summary = self._build_run_summary(state, won, cause_of_death, outcome)
        run_score_val = float(summary.get("run_score") or 0.0)

        run_outcome = {
            "won": won,
            "reward": run_score_val,
            "run_score": run_score_val,
            "floors_reached": self._max_floor_seen,
            "act_reached": self._max_act_seen,
            "avg_hp_pct_after_combat": summary.get("avg_hp_pct_after_combat"),
            "bosses_killed": summary.get("bosses_killed"),
        }

        for record in self._buffer:
            record["run_outcome"] = run_outcome

        self._flush_decisions()
        self._last_run_summary = summary
        self._enqueue_line(RUNS_PATH, summary)
        try:
            self._write_queue.join()
        except Exception:
            pass

        logger.info(
            "data_pipeline: ended run %s (%s, floor %s)",
            self.run_id,
            outcome,
            self._max_floor_seen,
        )

        self._run_active = False
        self.run_id = None

    def record_decision(
        self,
        state: dict,
        action: dict,
        reasoning: list[str] | str,
    ) -> None:
        """Record a state-action pair (call after action is sent)."""
        if not self._run_active or not self.run_id:
            return
        try:
            self._record_decision_inner(state, action, reasoning)
        except Exception as exc:
            logger.debug("data_pipeline record_decision failed: %s", exc)

    def _record_decision_inner(
        self,
        state: dict,
        action: dict,
        reasoning: list[str] | str,
    ) -> None:
        run = state.get("run") or {}
        state_type = str(state.get("state_type") or "")
        reason_text = (
            "; ".join(reasoning)
            if isinstance(reasoning, list)
            else str(reasoning)
        )

        immediate = self._immediate_reward(state, action)

        record = {
            "run_id": self.run_id,
            "timestamp": _utc_now(),
            "collector_instance": _collector_instance,
            "agent_version": self._agent_version,
            "floor": _safe_int(run.get("floor")),
            "act": _safe_int(run.get("act"), 1),
            "state_type": state_type,
            "state_snapshot": build_state_snapshot(
                state,
                enemy_intent_histories=self._enemy_intent_history,
            ),
            "action_taken": action,
            "action_reasoning": reason_text,
            "immediate_reward": immediate,
            "run_outcome": None,
        }

        self._buffer.append(record)
        idx = len(self._buffer) - 1
        self._decision_count += 1

        if state_type.lower() in COMBAT_TYPES:
            self._combat_decision_indices.append(idx)
            player = state.get("player") or {}
            self._combat_prev_hp = _safe_int(player.get("hp"))
            self._combat_prev_block = _safe_int(player.get("block"))
            self._combat_damage_since_decision = 0

    def _immediate_reward(self, state: dict, action: dict) -> float | dict | None:
        state_type = str(state.get("state_type") or "").lower()
        action_name = str(action.get("action") or "")

        if state_type in COMBAT_TYPES:
            return self._combat_immediate_reward(state)

        if state_type == "card_reward":
            if action_name == "skip_card_reward":
                return -1.0
            if action_name == "select_card_reward":
                return self._card_pick_reward(state, action)

        if state_type == "rest_site" and action_name == "choose_rest_option":
            return self._rest_option_reward(state, action)

        if state_type == "map" and action_name == "choose_map_node":
            return self._map_choice_reward(state, action)

        return None

    def _combat_immediate_reward(self, state: dict) -> dict[str, float | int]:
        from sts2_agent.scorer import combat_turn_shaping

        player = state.get("player") or {}
        hp = _safe_int(player.get("hp"))
        block = _safe_int(player.get("block"))

        hp_lost = 0
        if self._combat_prev_hp is not None and hp < self._combat_prev_hp:
            hp_lost = self._combat_prev_hp - hp

        block_gained = 0
        if self._combat_prev_block is not None and block > self._combat_prev_block:
            block_gained = block - self._combat_prev_block

        damage_dealt = self._combat_damage_since_decision
        shaping = combat_turn_shaping(hp_lost, block_gained, damage_dealt)

        return {
            "hp_lost_this_turn": hp_lost,
            "block_applied": block_gained,
            "damage_dealt": damage_dealt,
            "combat_score_contribution": shaping,
        }

    def _card_pick_reward(self, state: dict, action: dict) -> float | None:
        from sts2_agent.state_parse import extract_card_reward_cards

        cards = extract_card_reward_cards(state)
        idx = _safe_int(action.get("card_index"))
        if idx < len(cards):
            card = cards[idx]
        else:
            card = {}
        rarity = str(card.get("rarity") or "").lower()
        if "rare" in rarity:
            return 5.0
        if "uncommon" in rarity:
            return 2.0
        return 0.0

    def _rest_option_reward(self, state: dict, action: dict) -> float | None:
        from sts2_agent.state_parse import extract_rest_options, rest_option_index

        options = extract_rest_options(state)
        idx = _safe_int(action.get("index"))
        player = state.get("player") or {}
        hp = _safe_int(player.get("hp"))
        max_hp = max(_safe_int(player.get("max_hp"), 1), 1)
        ratio = hp / max_hp

        label = ""
        for list_idx, option in enumerate(options):
            if rest_option_index(option, list_idx) == idx:
                label = str(option.get("id") or option.get("name") or "").lower()
                break

        if any(k in label for k in ("rest", "heal", "sleep")) and ratio < 0.5:
            return 5.0
        if any(k in label for k in ("smith", "upgrade", "forge")) and ratio > 0.7:
            return 3.0
        return None

    def _map_choice_reward(self, state: dict, action: dict) -> float | None:
        from sts2_agent.state_parse import extract_map_choices, map_choice_room_type

        choices = extract_map_choices(state)
        idx = _safe_int(action.get("index"))
        for opt in choices:
            opt_idx = _safe_int(opt.get("index"), -1)
            if opt_idx == idx:
                room = map_choice_room_type(opt).lower()
                if "rest" in room:
                    return 3.0
                break
        return None

    def _begin_combat(self, state: dict) -> None:
        from sts2_agent.enemy_compendium import begin_combat_observation

        begin_combat_observation()
        self._enemy_intent_history = {}
        player = state.get("player") or {}
        self._combat_state_type = str(state.get("state_type") or "").lower()
        self._in_combat = True
        self._combat_start_hp = _safe_int(player.get("hp"))
        self._combat_start_max_hp = max(_safe_int(player.get("max_hp"), 1), 1)
        self._combat_enemy_ids = set()
        self._combat_decision_indices = []
        self._enemy_hp_snapshot = {}
        self._combat_prev_hp = _safe_int(player.get("hp"))
        self._combat_prev_block = _safe_int(player.get("block"))
        self._combat_damage_since_decision = 0
        for enemy in (state.get("battle") or {}).get("enemies") or []:
            if isinstance(enemy, dict) and enemy.get("entity_id"):
                eid = str(enemy["entity_id"])
                self._combat_enemy_ids.add(eid)
                self._enemy_hp_snapshot[eid] = _safe_int(enemy.get("hp"))
        self._track_enemy_intent_history(state)

    def _intent_history_signature(self, entry: dict) -> str:
        return (
            f"{entry.get('intent')}|{entry.get('damage')}|{entry.get('block')}|"
            f"{','.join(entry.get('tags') or [])}"
        )

    def _track_enemy_intent_history(self, state: dict) -> None:
        """Append distinct enemy intents (deduped per poll) - up to 3 per entity per combat."""
        from sts2_agent.enemy_compendium import compact_enemy_intent

        for enemy in (state.get("battle") or {}).get("enemies") or []:
            if not isinstance(enemy, dict):
                continue
            if int(enemy.get("hp") or 0) <= 0:
                continue
            living = [
                e
                for e in (state.get("battle") or {}).get("enemies") or []
                if isinstance(e, dict) and int(e.get("hp") or 0) > 0
            ]
            compact = compact_enemy_intent(enemy, peers=living)
            if not compact:
                continue
            entity_id = str(enemy.get("entity_id") or enemy.get("id") or "")
            if not entity_id:
                continue
            entry = {
                "entity_id": entity_id,
                "intent": compact.get("intent"),
                "damage": compact.get("damage"),
                "block": compact.get("block"),
                "tags": list(compact.get("tags") or []),
            }
            history = self._enemy_intent_history.setdefault(entity_id, [])
            if history and self._intent_history_signature(history[-1]) == self._intent_history_signature(
                entry
            ):
                continue
            history.append(entry)
            if len(history) > ENEMY_INTENT_HISTORY_LEN:
                self._enemy_intent_history[entity_id] = history[-ENEMY_INTENT_HISTORY_LEN:]

    def _track_enemy_damage(self, state: dict) -> None:
        """Accumulate damage dealt from enemy HP drops between state polls."""
        for enemy in (state.get("battle") or {}).get("enemies") or []:
            if not isinstance(enemy, dict):
                continue
            eid = str(enemy.get("entity_id") or "")
            if not eid:
                continue
            hp = _safe_int(enemy.get("hp"))
            prev = self._enemy_hp_snapshot.get(eid)
            if prev is not None and hp < prev:
                dealt = prev - hp
                self._damage_dealt_run += dealt
                self._combat_damage_since_decision += dealt
            self._enemy_hp_snapshot[eid] = hp

    def _end_combat(self, state: dict) -> None:
        from sts2_agent.potions import get_potion_drop_tracker
        from sts2_agent.scorer import combat_reward

        get_potion_drop_tracker().note_combat_ended(self._combat_state_type)

        # Credit damage for enemies removed from battle (killed)
        living_ids: set[str] = set()
        for enemy in (state.get("battle") or {}).get("enemies") or []:
            if isinstance(enemy, dict) and enemy.get("entity_id"):
                living_ids.add(str(enemy["entity_id"]))
        for eid, prev_hp in list(self._enemy_hp_snapshot.items()):
            if eid not in living_ids and prev_hp > 0:
                dealt = prev_hp
                self._damage_dealt_run += dealt
                self._combat_damage_since_decision += dealt

        player = state.get("player") or {}
        end_hp = _safe_int(player.get("hp"), self._last_hp or 0)
        start_hp = (
            self._combat_start_hp
            if self._combat_start_hp is not None
            else end_hp
        )
        max_hp = self._combat_start_max_hp or max(
            _safe_int(player.get("max_hp"), 1), 1
        )
        won_combat = end_hp > 0
        reward = combat_reward(start_hp, end_hp, max_hp, won_combat)

        self._hp_before_each_combat.append(start_hp)
        self._hp_after_each_combat.append(end_hp)
        self._max_hp_each_combat.append(max_hp)
        self._combat_rewards.append(float(reward))

        if self._combat_state_type == "boss" and won_combat:
            self._bosses_killed += 1

        if self._combat_decision_indices:
            self._apply_reward_to_indices(self._combat_decision_indices, reward)

        from sts2_agent.enemy_compendium import finalize_combat_observation

        finalize_combat_observation(state)

        self._in_combat = False
        self._combat_decision_indices = []
        self._combat_enemy_ids = set()
        self._enemy_hp_snapshot = {}
        self._enemy_intent_history = {}
        self._combat_prev_hp = None
        self._combat_prev_block = None
        self._combat_damage_since_decision = 0

    def _apply_reward_to_indices(self, indices: list[int], total_reward: float) -> None:
        if not indices:
            return
        buf = list(self._buffer)
        per = total_reward / len(indices)
        for idx in indices:
            if 0 <= idx < len(buf):
                existing = buf[idx].get("immediate_reward")
                if isinstance(existing, dict):
                    updated = dict(existing)
                    updated["combat_end_reward"] = per
                    updated["combat_score_contribution"] = float(
                        updated.get("combat_score_contribution", 0)
                    ) + per
                    buf[idx]["immediate_reward"] = updated
                elif existing is None:
                    buf[idx]["immediate_reward"] = {
                        "combat_end_reward": per,
                        "combat_score_contribution": per,
                    }
                else:
                    buf[idx]["immediate_reward"] = {
                        "combat_end_reward": per,
                        "combat_score_contribution": float(existing) + per,
                    }
        self._buffer = deque(buf, maxlen=MAX_BUFFER)

    def _flush_decisions(self) -> None:
        for record in self._buffer:
            self._enqueue_line(DECISIONS_PATH, record)
        self._buffer.clear()

    def _combat_hp_percentages(self) -> tuple[float, float, float]:
        """Return (avg, best, worst) HP% remaining after each combat."""
        pcts: list[float] = []
        for after, mx in zip(self._hp_after_each_combat, self._max_hp_each_combat):
            if mx > 0:
                pcts.append(after / mx)
        if not pcts:
            return 0.0, 0.0, 0.0
        return sum(pcts) / len(pcts), max(pcts), min(pcts)

    def _build_run_summary(
        self,
        state: dict | None,
        won: bool,
        cause_of_death: str | None,
        outcome: str,
    ) -> dict:
        # run_score: floors*15 + (act-1)*60 + avg_hp_pct*100 + win(1000) + bosses*100
        from sts2_agent.scorer import run_score

        player = (state or {}).get("player") or {}
        deck = list(self._last_deck)
        if not deck and state:
            deck = extract_deck_card_ids(state)

        relics = list(self._last_relics)
        if not relics and state:
            relics = extract_relic_ids(state)

        potions_at_death = extract_potion_ids(state) if state else []

        avg_hp_pct, best_hp_pct, worst_hp_pct = self._combat_hp_percentages()

        run_data = {
            "floors_reached": self._max_floor_seen,
            "act_reached": self._max_act_seen,
            "avg_hp_pct_after_combat": avg_hp_pct,
            "final_deck": deck,
            "potions_at_death": potions_at_death,
            "bosses_killed": self._bosses_killed,
            "won": won,
        }
        run_score_val = run_score(run_data)

        ended_at = datetime.now(timezone.utc)
        started_at = self._run_started_at or ended_at
        duration_sec = max(0.0, (ended_at - started_at).total_seconds())

        return {
            "run_id": self.run_id,
            "timestamp": ended_at.isoformat(),
            "started_at": started_at.isoformat(),
            "run_duration_sec": round(duration_sec, 1),
            "collector_instance": _collector_instance,
            "source": "agent",
            "agent_version": self._agent_version,
            "character": self._character,
            "ascension": self._ascension,
            "won": won,
            "floors_reached": self._max_floor_seen,
            "act_reached": self._max_act_seen,
            "cause_of_death": cause_of_death,
            "final_deck": deck,
            "final_relics": [r for r in relics if r],
            "total_decisions": self._decision_count,
            "total_damage_taken": self._damage_taken_run,
            "total_damage_dealt": self._damage_dealt_run,
            "gold_at_death": _safe_int(player.get("gold")),
            "outcome": outcome,
            "run_score": run_score_val,
            "avg_hp_pct_after_combat": avg_hp_pct,
            "best_combat_hp_pct": best_hp_pct,
            "worst_combat_hp_pct": worst_hp_pct,
            "bosses_killed": self._bosses_killed,
            "potions_at_death": potions_at_death,
            "hp_before_each_combat": list(self._hp_before_each_combat),
            "hp_after_each_combat": list(self._hp_after_each_combat),
            "combat_rewards": list(self._combat_rewards),
        }

    def _infer_death_cause(self, state_type: str, state: dict) -> str | None:
        if state_type != "game_over" and _safe_int((state.get("player") or {}).get("hp")) > 0:
            return None
        battle = state.get("battle") or {}
        enemies = [e.get("name") for e in battle.get("enemies") or [] if isinstance(e, dict)]
        room = state_type
        if state_type in ("monster", "elite", "boss"):
            room = f"{state_type} combat"
        if enemies:
            return f"{room} - vs {', '.join(str(e) for e in enemies[:2])} - hp reached 0"
        return f"{room} - hp reached 0"

    def flush_on_exit(self) -> None:
        """Best-effort flush when agent stops unexpectedly."""
        try:
            if self._run_active and self._buffer:
                if self._in_combat:
                    self._end_combat({})
                summary = self._build_run_summary(
                    None, won=False, cause_of_death="agent interrupted", outcome="lost"
                )
                run_score_val = float(summary.get("run_score") or 0.0)
                outcome = {
                    "won": False,
                    "reward": run_score_val,
                    "run_score": run_score_val,
                    "interrupted": True,
                }
                for record in self._buffer:
                    record["run_outcome"] = outcome
                self._flush_decisions()
                self._enqueue_line(RUNS_PATH, summary)
                self._run_active = False
            self._write_queue.join()
        except Exception as exc:
            logger.debug("data_pipeline flush_on_exit failed: %s", exc)


_pipeline: DataPipeline | None = None


def get_pipeline() -> DataPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline()
    return _pipeline


def set_agent_version(version: str) -> None:
    """Tag subsequent runs/decisions (e.g. rules_v1 vs bc_v1_64runs)."""
    get_pipeline()._agent_version = str(version)


def get_last_run_summary() -> dict | None:
    """Most recently ended run summary (set on game_over)."""
    return get_pipeline()._last_run_summary


def observe_state(state: dict) -> None:
    get_pipeline().observe_state(state)


def record_decision(state: dict, action: dict, reasoning: list[str] | str) -> None:
    get_pipeline().record_decision(state, action, reasoning)


def record_handler_decision(
    state: dict,
    action: dict | None,
    reasoning: list[str],
    *,
    handler: str,
) -> None:
    """Convenience wrapper for handler modules."""
    if action is None:
        return
    merged = reasoning if reasoning else [f"{handler} decision"]
    record_decision(state, action, merged)
