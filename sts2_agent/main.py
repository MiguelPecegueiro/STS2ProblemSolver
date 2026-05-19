"""STS2 autonomous agent - poll game state and send rule-based actions."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import requests

from sts2_agent.agent import (
    configure_card_reward_bc,
    configure_policy,
    configure_ppo_macro,
    decide,
    state_fingerprint,
)
from sts2_agent.api import DEFAULT_BASE_URL, STS2APIError, STS2Client
from sts2_agent import combat, event, map as map_handler, rewards, rest, shop
from sts2_agent.data_pipeline import (
    configure_data_paths,
    get_pipeline,
    observe_state,
    set_agent_version,
    set_game_version,
)
from sts2_agent.decision_log import log_decision, setup_decision_logging
from sts2_agent.knowledge import load_knowledge, refresh_cache
from sts2_agent.qwen_advisor import configure_qwen, get_qwen_advisor
from sts2_agent.graceful_shutdown import (
    GracefulShutdown,
    SingleRunController,
    install_graceful_shutdown_handler,
    shutdown_help_message,
)
from sts2_agent.menu import IN_RUN_STATE_TYPES, MenuFlow
from sts2_agent.state_parse import (
    extract_shop_items,
    is_card_select_active,
    treasure_can_proceed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARD_REWARD_MODEL = REPO_ROOT / "models" / "bc_human_card.pt"
DEFAULT_CARD_REWARD_CONFIG = REPO_ROOT / "models" / "bc_human_card_config.json"
DEFAULT_PPO_MACRO_MODEL = REPO_ROOT / "models" / "ppo_v5.pt"
DEFAULT_PPO_MACRO_CONFIG = REPO_ROOT / "models" / "ppo_config.json"

POLL_INTERVAL_SEC = 0.5
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

# Logged on each run/decision row - bump when rules or model checkpoint changes.
AGENT_VERSION_RULES = "rules_v1"
AGENT_VERSION = "ppo_v6"

# Qwen (LM Studio) — runtime only; fails open if API unavailable.
QWEN_ENABLED = True
QWEN_COMBAT_ENABLED = False
QWEN_MACRO_ENABLED = False
QWEN_MACRO_CONTEXT_ENABLED = True
CARD_REWARD_BC_ENABLED = True
PPO_MACRO_ENABLED = True
QWEN_URL = "http://127.0.0.1:1234/v1/chat/completions"
QWEN_MODEL = "qwen3-4b-instruct-2507"
QWEN_TIMEOUT = 10
QWEN_LOG_FULL_PROMPT = True


def _record_training_data(state: dict, action: dict, reasons: list[str]) -> None:
    state_type = str(state.get("state_type") or "").lower()
    try:
        if state_type in combat.COMBAT_STATE_TYPES or state_type == "hand_select":
            combat.record_training(state, action, reasons)
        elif state_type == "map":
            map_handler.record_training(state, action, reasons)
        elif state_type in ("card_reward", "rewards", "treasure"):
            rewards.record_training(state, action, reasons)
        elif state_type == "rest_site":
            rest.record_training(state, action, reasons)
        elif state_type in ("shop", "fake_merchant"):
            shop.record_training(state, action, reasons)
        elif state_type == "event":
            event.record_training(state, action, reasons)
        elif is_card_select_active(state):
            from sts2_agent import card_select

            card_select.record_training(state, action, reasons)
    except Exception:
        pass


def _run_in_progress(state: dict) -> bool:
    state_type = str(state.get("state_type") or "").lower()
    if state_type in IN_RUN_STATE_TYPES:
        return True
    run = state.get("run") or {}
    return bool(run.get("floor")) and state_type not in ("menu", "game_over", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slay the Spire 2 rule-based agent")
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument(
        "--url",
        default=None,
        help=f"STS2MCP API base URL (default: {DEFAULT_BASE_URL})",
    )
    endpoint.add_argument(
        "--port",
        type=int,
        default=None,
        help="STS2MCP listen port (builds http://127.0.0.1:<port>; mutually exclusive with --url)",
    )
    parser.add_argument(
        "--instance-id",
        default=None,
        help="Parallel collector instance id (tags records; disables compendium disk writes)",
    )
    parser.add_argument(
        "--abandon-stale-run",
        action="store_true",
        help=(
            "On main menu with continue/abandon but no singleplayer, abandon the "
            "in-progress save and start fresh (default when --instance-id is set)"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for decisions.jsonl and runs.jsonl (default: data/ or data/instances/<id>)",
    )
    parser.add_argument(
        "--game-version",
        default=None,
        metavar="ID",
        help=(
            "Balance patch tag for runs.jsonl / decisions.jsonl "
            "(default: STS2_GAME_VERSION env or 'unknown')"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=POLL_INTERVAL_SEC,
        help="Seconds between state polls",
    )
    parser.add_argument(
        "--character",
        default="ironclad",
        help="Character to pick on new runs (default: ironclad)",
    )
    parser.add_argument(
        "--refresh-knowledge",
        action="store_true",
        help="Force refresh Spire Codex / community stats cache",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    parser.add_argument(
        "--no-compendium",
        action="store_true",
        help=(
            "Record enemy intents for training data but do not use learned patterns "
            "in combat (no enrich_incoming_damage / debuff pressure)"
        ),
    )
    parser.add_argument(
        "--policy",
        action="store_true",
        help=(
            "Use behavioral-cloning policy (models/policy_net.pt) for all screens; "
            "fall back to rules when the prediction is invalid"
        ),
    )
    parser.add_argument(
        "--policy-combat-only",
        action="store_true",
        help=(
            "Use BC policy only in combat (monster/elite/boss/hand_select); "
            "rules elsewhere. Ignored if --policy is set"
        ),
    )
    parser.add_argument(
        "--combat-policy",
        action="store_true",
        help=(
            "With --policy: policy-first in combat (legacy). Default is planner-first; "
            "policy only when the combat planner abstains."
        ),
    )
    parser.add_argument(
        "--card-reward-model",
        type=Path,
        default=DEFAULT_CARD_REWARD_MODEL,
        help="Human BC checkpoint for card_reward screens (default: models/bc_human_card.pt)",
    )
    parser.add_argument(
        "--card-reward-config",
        type=Path,
        default=DEFAULT_CARD_REWARD_CONFIG,
        help="Config for --card-reward-model (default: models/bc_human_card_config.json)",
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=DEFAULT_PPO_MACRO_MODEL,
        help="PPO checkpoint for map/shop/rest/event (default: models/ppo_v5.pt)",
    )
    parser.add_argument(
        "--ppo-config",
        type=Path,
        default=DEFAULT_PPO_MACRO_CONFIG,
        help="Config for --ppo-model (default: models/ppo_config.json)",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Play one full run then exit (no continuous collection loop)",
    )
    return parser.parse_args()


def run(
    client: STS2Client,
    interval: float,
    menu_flow: MenuFlow,
    graceful: GracefulShutdown,
    *,
    single_run: bool = False,
) -> None:
    last_fingerprint: str | None = None
    last_action_key: str | None = None
    last_potion_action: tuple[object, ...] | None = None
    single = SingleRunController() if single_run else None

    if single_run:
        logging.info("Single-run mode: will exit after one completed run")
    logging.info("Autonomous loop active - %s", shutdown_help_message())

    while True:
        try:
            state = client.get_state()
        except STS2APIError as exc:
            logging.warning("API error: %s", exc)
            time.sleep(interval)
            continue
        except requests.ConnectionError:
            logging.info("Waiting for game API at %s ...", client.base_url)
            time.sleep(interval * 2)
            continue

        observe_state(state)
        state_type = str(state.get("state_type") or "?")
        in_run = _run_in_progress(state)
        graceful.observe(state, run_in_progress=in_run)
        if graceful.should_exit(state, run_in_progress=in_run):
            logging.info("Graceful shutdown complete")
            break

        if _run_in_progress(state):
            menu_flow.on_run_started(state)

        if menu_flow.should_handle(state):
            decision_action, decision_reasons = menu_flow.decide(state)
            fingerprint = f"menu_flow|{state_type}|{menu_flow.wait_until}"
            skip_dedup = True
        else:
            fingerprint = state_fingerprint(state)
            decision = decide(state)
            decision_action = decision.action
            decision_reasons = decision.reasons
            skip_dedup = False

        if decision_action is None:
            last_fingerprint = fingerprint
            time.sleep(interval)
            continue

        action_key = f"{fingerprint}|{decision_action}"
        stuck_leaving_shop = (
            not skip_dedup
            and str(state_type) in ("shop", "fake_merchant")
            and isinstance(decision_action, dict)
            and decision_action.get("action") == "proceed"
            and not extract_shop_items(state)
        )
        stuck_rewards = (
            not skip_dedup
            and str(state_type) == "rewards"
            and isinstance(decision_action, dict)
            and decision_action.get("action") in ("proceed", "claim_reward", "discard_potion")
        )
        stuck_treasure = (
            not skip_dedup
            and str(state_type) == "treasure"
            and isinstance(decision_action, dict)
            and decision_action.get("action") == "proceed"
            and treasure_can_proceed(state)
        )
        stuck_card_select = (
            not skip_dedup
            and is_card_select_active(state)
            and isinstance(decision_action, dict)
            and decision_action.get("action") in ("select_card", "confirm_selection")
        )
        potion_action_key: tuple[object, ...] | None = None
        if isinstance(decision_action, dict) and decision_action.get("action") == "use_potion":
            potion_action_key = (
                "use_potion",
                decision_action.get("slot"),
                decision_action.get("target"),
            )
        stuck_potion = (
            not skip_dedup
            and potion_action_key is not None
            and potion_action_key == last_potion_action
        )
        if (
            not skip_dedup
            and action_key == last_action_key
            and fingerprint == last_fingerprint
            and not stuck_leaving_shop
            and not stuck_rewards
            and not stuck_card_select
            and not stuck_treasure
            and not stuck_potion
        ):
            time.sleep(interval)
            continue

        logging.info("state=%s action=%s", state_type, decision_action)
        log_decision(str(state_type), decision_action, decision_reasons)

        try:
            new_state = client.send_action(decision_action)
        except STS2APIError as exc:
            logging.error("Action rejected: %s - %s", decision_action, exc)
            if (
                isinstance(decision_action, dict)
                and decision_action.get("action") in (
                    "select_card",
                    "confirm_selection",
                    "cancel_selection",
                )
            ):
                from sts2_agent.card_select import note_card_select_action_failed

                note_card_select_action_failed(state, decision_action)
                if decision_action.get("action") == "confirm_selection":
                    last_action_key = None
            elif (
                isinstance(decision_action, dict)
                and decision_action.get("action")
                in ("combat_select_card", "combat_confirm_selection")
            ):
                from sts2_agent.combat import note_hand_select_action_failed

                note_hand_select_action_failed(state, decision_action)
                last_action_key = None
            elif (
                isinstance(decision_action, dict)
                and decision_action.get("action") == "use_potion"
                and decision_action.get("slot") is not None
            ):
                from sts2_agent.potions import mark_potion_use_failed

                mark_potion_use_failed(
                    state.get("player") or {},
                    int(decision_action["slot"]),
                )
                last_action_key = None
                if potion_action_key is not None:
                    last_potion_action = potion_action_key
            elif (
                isinstance(decision_action, dict)
                and decision_action.get("action") == "choose_event_option"
                and decision_action.get("index") is not None
            ):
                from sts2_agent.event import mark_event_option_failed

                mark_event_option_failed(state, int(decision_action["index"]))
                last_action_key = None
            elif (
                isinstance(decision_action, dict)
                and decision_action.get("action") == "choose_map_node"
                and decision_action.get("index") is not None
            ):
                from sts2_agent.map import mark_map_choice_failed

                mark_map_choice_failed(state, int(decision_action["index"]))
                last_action_key = None
            elif (
                isinstance(decision_action, dict)
                and str(state_type) == "card_reward"
                and decision_action.get("action") in ("proceed", "skip_card_reward")
            ):
                last_action_key = None
        else:
            if isinstance(decision_action, dict) and decision_action.get("action") in (
                "select_card",
                "confirm_selection",
                "cancel_selection",
            ):
                from sts2_agent.card_select import sync_card_select_after_action

                sync_card_select_after_action(state, new_state, decision_action)
            elif isinstance(decision_action, dict) and decision_action.get("action") in (
                "combat_select_card",
                "combat_confirm_selection",
            ):
                from sts2_agent.combat import sync_hand_select_after_action

                sync_hand_select_after_action(state, new_state, decision_action)
            elif isinstance(decision_action, dict) and str(state_type) == "event":
                from sts2_agent.event import clear_event_session

                if str(new_state.get("state_type") or "").lower() != "event":
                    clear_event_session(state)
            elif isinstance(decision_action, dict) and str(state_type) == "map":
                from sts2_agent.map import clear_map_session

                if str(new_state.get("state_type") or "").lower() != "map":
                    clear_map_session(state)
            elif isinstance(decision_action, dict):
                action_name = str(decision_action.get("action") or "")
                if action_name == "claim_reward" and str(state_type) == "rewards":
                    from sts2_agent.rewards import note_card_reward_claimed

                    note_card_reward_claimed(state, int(decision_action.get("index", -1)))
                elif action_name == "skip_card_reward" and str(state_type) == "card_reward":
                    from sts2_agent.rewards import note_card_reward_skipped

                    note_card_reward_skipped()
                elif action_name == "select_card_reward" and str(state_type) == "card_reward":
                    from sts2_agent.rewards import note_card_reward_selected

                    note_card_reward_selected()
                elif action_name == "proceed" and str(state_type) == "rewards":
                    from sts2_agent.rewards import note_rewards_screen_done

                    note_rewards_screen_done()
                elif action_name == "use_potion" and decision_action.get("slot") is not None:
                    from sts2_agent.potions import note_potion_use_no_effect

                    if note_potion_use_no_effect(state, new_state, decision_action):
                        last_potion_action = None
            if not skip_dedup:
                _record_training_data(state, decision_action, decision_reasons)
            last_fingerprint = fingerprint
            last_action_key = action_key
            if potion_action_key is not None:
                last_potion_action = potion_action_key

        time.sleep(interval)


def _resolve_base_url(args: argparse.Namespace) -> str:
    if args.port is not None:
        return f"http://127.0.0.1:{args.port}"
    return args.url or DEFAULT_BASE_URL


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
    )
    setup_decision_logging()

    if args.data_dir is not None or args.instance_id is not None:
        out_dir = configure_data_paths(args.data_dir, args.instance_id)
        logging.info("Training data directory: %s", out_dir)

    game_version = set_game_version(args.game_version)
    logging.info("Game version tag: %s", game_version)
    if args.instance_id is not None and not args.no_compendium:
        args.no_compendium = True
        logging.info(
            "Parallel instance %s: compendium disk writes disabled (use shared merge pass)",
            args.instance_id,
        )

    abandon_stale = args.abandon_stale_run or args.instance_id is not None
    if abandon_stale:
        logging.info(
            "Menu: will abandon stale in-progress saves to reach singleplayer "
            "(shared profile - use separate profiles per instance when possible)"
        )

    logging.info("Loading knowledge base...")
    try:
        if args.refresh_knowledge:
            refresh_cache(force=True)
        load_knowledge(force_refresh=args.refresh_knowledge)
        configure_qwen(
            enabled=QWEN_ENABLED,
            combat_enabled=QWEN_COMBAT_ENABLED,
            macro_enabled=QWEN_MACRO_ENABLED,
            macro_context_enabled=QWEN_MACRO_CONTEXT_ENABLED,
            url=QWEN_URL,
            model=QWEN_MODEL,
            timeout=float(QWEN_TIMEOUT),
            log_full_prompt=QWEN_LOG_FULL_PROMPT,
        )
        get_qwen_advisor().reload_expert_knowledge()
        logging.info("Knowledge base ready")
    except requests.RequestException as exc:
        logging.error(
            "Failed to load knowledge (need network for first run): %s", exc
        )
        return 1

    client = STS2Client(base_url=_resolve_base_url(args))
    menu_flow = MenuFlow(
        character=args.character,
        abandon_stale_runs=abandon_stale,
    )
    graceful = GracefulShutdown()

    def _request_graceful_shutdown() -> None:
        graceful.request()
        menu_flow.block_restart()

    install_graceful_shutdown_handler(_request_graceful_shutdown)

    if args.no_compendium:
        combat.set_compendium_decisions_enabled(False)
        logging.info(
            "Enemy compendium: observation/logging only (combat uses live API intents)"
        )

    card_bc_on = False
    if CARD_REWARD_BC_ENABLED:
        if args.card_reward_model.exists() and args.card_reward_config.exists():
            configure_card_reward_bc(
                enabled=True,
                model_path=args.card_reward_model,
                config_path=args.card_reward_config,
            )
            try:
                from training.card_reward_bc import get_card_reward_policy

                get_card_reward_policy(
                    args.card_reward_model,
                    args.card_reward_config,
                )
                card_bc_on = True
                logging.info(
                    "Card reward BC: %s (Qwen skipped on card_reward)",
                    args.card_reward_model,
                )
            except Exception as exc:
                logging.error("Failed to load card reward BC model: %s", exc)
                configure_card_reward_bc(enabled=False)
                return 1
        else:
            logging.warning(
                "Card reward BC enabled but checkpoint missing (%s / %s) - using rules/Qwen",
                args.card_reward_model,
                args.card_reward_config,
            )
            configure_card_reward_bc(enabled=False)
    else:
        configure_card_reward_bc(enabled=False)

    ppo_macro_on = False
    if PPO_MACRO_ENABLED:
        from training.ppo_macro import resolve_ppo_macro_paths

        model_arg = args.ppo_model if args.ppo_model.exists() else None
        cfg_arg = args.ppo_config if args.ppo_config.exists() else None
        ppo_model, ppo_cfg = resolve_ppo_macro_paths(model_arg, cfg_arg)
        if ppo_model.exists() and ppo_cfg.exists():
            configure_ppo_macro(
                enabled=True,
                model_path=ppo_model,
                config_path=ppo_cfg,
            )
            try:
                from training.ppo_macro import get_ppo_macro_policy

                get_ppo_macro_policy(ppo_model, ppo_cfg)
                ppo_macro_on = True
                logging.info(
                    "PPO macro: %s for map/shop/rest/event (Qwen context=%s)",
                    ppo_model,
                    QWEN_MACRO_CONTEXT_ENABLED,
                )
            except Exception as exc:
                logging.error("Failed to load PPO macro model: %s", exc)
                configure_ppo_macro(enabled=False)
                return 1
        else:
            logging.warning(
                "PPO macro enabled but checkpoint missing (%s) - rules for macro screens",
                args.ppo_model,
            )
            configure_ppo_macro(enabled=False)
    else:
        configure_ppo_macro(enabled=False)

    use_policy = args.policy or args.policy_combat_only
    use_neural = use_policy or card_bc_on or ppo_macro_on
    set_agent_version(AGENT_VERSION if use_neural else AGENT_VERSION_RULES)

    if use_policy:
        configure_policy(
            enabled=args.policy,
            combat_only=args.policy_combat_only,
            no_combat_policy=not args.combat_policy,
        )
        try:
            from training.inference import get_policy

            get_policy()
        except Exception as exc:
            logging.error("Failed to load policy model: %s", exc)
            return 1
        combat_routing = (
            "planner first, policy on abstain"
            if not args.combat_policy
            else "policy first, rules fallback"
        )
        if args.policy:
            logging.info(
                "Policy mode: model for non-combat screens; combat=%s",
                combat_routing,
            )
        else:
            logging.info(
                "Policy mode: combat only (%s); rules elsewhere",
                combat_routing,
            )

    active_version = AGENT_VERSION if use_neural else AGENT_VERSION_RULES
    logging.info(
        "STS2 agent started (url=%s, poll every %.2fs, character=%s, agent_version=%s, "
        "card_bc=%s, ppo_macro=%s)",
        client.base_url,
        args.interval,
        args.character,
        active_version,
        card_bc_on,
        ppo_macro_on,
    )
    get_pipeline()
    try:
        run(client, args.interval, menu_flow, graceful, single_run=args.single_run)
        get_pipeline().flush_on_exit()
        return 0
    except KeyboardInterrupt:
        logging.info("Stopped immediately (Ctrl+C)")
        get_pipeline().flush_on_exit()
        return 0


if __name__ == "__main__":
    sys.exit(main())
