"""Menu / game-over flow - auto-restart runs until Ctrl+C."""

from __future__ import annotations

import logging
import time
from typing import Any

from sts2_agent.data_pipeline import get_last_run_summary

logger = logging.getLogger(__name__)

POST_GAME_OVER_DELAY_SEC = 2.0
POST_MAIN_MENU_DELAY_SEC = 2.0
POST_ABANDON_DELAY_SEC = 2.0
DEFAULT_CHARACTER = "ironclad"

IN_RUN_STATE_TYPES = frozenset(
    {
        "map",
        "monster",
        "elite",
        "boss",
        "event",
        "shop",
        "fake_merchant",
        "rest_site",
        "rewards",
        "card_reward",
        "treasure",
        "card_select",
        "hand_select",
    }
)


def _menu_option_string(raw: Any) -> str | None:
    """Extract the API menu_select option id (e.g. IRONCLAD, confirm)."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        if raw.get("enabled") is False:
            return None
        for key in ("name", "id", "option", "value"):
            val = raw.get(key)
            if val is not None and str(val).strip():
                return str(val)
        return None
    text = str(raw).strip()
    return text or None


def _collect_menu_options(raw_list: list) -> list[str]:
    out: list[str] = []
    for item in raw_list:
        opt = _menu_option_string(item)
        if opt is not None:
            out.append(opt)
    return out


def extract_menu_options(state: dict) -> list[str]:
    """Advertised menu_select options from menu or game_over screens."""
    raw = state.get("options")
    if isinstance(raw, list):
        return _collect_menu_options(raw)

    screen = state.get("menu")
    if isinstance(screen, dict) and isinstance(screen.get("options"), list):
        return _collect_menu_options(screen["options"])

    game_over = state.get("game_over")
    if isinstance(game_over, dict) and isinstance(game_over.get("options"), list):
        return _collect_menu_options(game_over["options"])

    return []


def get_menu_screen(state: dict) -> str:
    return str(state.get("menu_screen") or "").lower()


def log_run_summary(state: dict) -> list[str]:
    """Log the ended run to the console."""
    summary = get_last_run_summary()
    reasons: list[str] = []

    if summary:
        outcome = "WIN" if summary.get("won") else "LOSS"
        logger.info(
            "Run ended - %s | %s | score %.0f | floor %s act %s | "
            "HP conserv %.0f%% | bosses %s | decisions %s | gold %s",
            outcome,
            summary.get("character", "?"),
            float(summary.get("run_score") or 0),
            summary.get("floors_reached", "?"),
            summary.get("act_reached", "?"),
            float(summary.get("avg_hp_pct_after_combat") or 0) * 100,
            summary.get("bosses_killed", 0),
            summary.get("total_decisions", "?"),
            summary.get("gold_at_death", "?"),
        )
        if summary.get("cause_of_death"):
            logger.info("Cause of death: %s", summary["cause_of_death"])
        reasons.append(
            f"run summary: {outcome} floor {summary.get('floors_reached')} "
            f"({summary.get('character', '?')})"
        )
    else:
        run = state.get("run") or {}
        player = state.get("player") or {}
        logger.info(
            "Run ended - floor %s act %s HP %s/%s gold %s",
            run.get("floor", "?"),
            run.get("act", "?"),
            player.get("hp", "?"),
            player.get("max_hp", "?"),
            player.get("gold", "?"),
        )
        reasons.append("run ended (no pipeline summary yet)")

    return reasons


def _normalize_option(value: str) -> str:
    return str(value).lower().replace(" ", "_").replace("-", "_")


def _pick_option(options: list[str], *needles: str) -> str | None:
    """Return the first API option string matching any needle."""
    for opt in options:
        norm = _normalize_option(opt)
        for needle in needles:
            n = _normalize_option(needle)
            if norm == n or n in norm or norm in n:
                return opt
    return None


def _character_ids_match(selected: str, wanted: str) -> bool:
    """True when API character id/name matches the configured character."""
    a = _normalize_option(selected)
    b = _normalize_option(wanted)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    aliases = {
        "ironclad": {"ironclad", "the_ironclad"},
        "silent": {"silent", "the_silent"},
        "defect": {"defect", "the_defect"},
        "necrobinder": {"necrobinder", "the_necrobinder"},
        "regent": {"regent", "the_regent"},
    }
    for group in aliases.values():
        if a in group and b in group:
            return True
    return False


def _selected_character(state: dict) -> str | None:
    """Currently highlighted character on character_select, if exposed."""
    for key in ("selected_character", "character_id", "current_character"):
        val = state.get(key)
        if val:
            return str(val)

    menu = state.get("menu")
    if isinstance(menu, dict):
        for key in ("selected_character", "character_id"):
            val = menu.get(key)
            if val:
                return str(val)

    chars = state.get("characters")
    if not isinstance(chars, list) and isinstance(menu, dict):
        chars = menu.get("characters")
    if isinstance(chars, list):
        for entry in chars:
            if not isinstance(entry, dict):
                continue
            if entry.get("selected") or entry.get("is_selected") or entry.get("highlighted"):
                return str(
                    entry.get("character_id")
                    or entry.get("name")
                    or entry.get("id")
                    or ""
                )
    return None


class MenuFlow:
    """State machine: game_over → main menu → singleplayer → Ironclad → new run."""

    def __init__(
        self,
        character: str = DEFAULT_CHARACTER,
        *,
        abandon_stale_runs: bool = False,
    ) -> None:
        self.character = character
        self.abandon_stale_runs = abandon_stale_runs
        self._active = False
        self._wait_until = 0.0
        self._logged_summary = False
        self._sent_main_menu = False
        self._character_picked = False
        self._block_restart = False
        self._awaiting_abandon_confirm = False

    def reset(self) -> None:
        self._active = False
        self._wait_until = 0.0
        self._logged_summary = False
        self._sent_main_menu = False
        self._character_picked = False
        self._awaiting_abandon_confirm = False

    def block_restart(self) -> None:
        """After game over, do not navigate main menu to start another run."""
        self._block_restart = True

    @property
    def wait_until(self) -> float:
        return self._wait_until

    def should_handle(self, state: dict) -> bool:
        state_type = str(state.get("state_type") or "").lower()
        if self._block_restart and state_type == "menu":
            return False
        if state_type == "game_over":
            return True
        if state_type == "menu":
            return True
        if self._active:
            return True
        return False

    def on_run_started(self, state: dict) -> None:
        """Called when we detect an in-run screen - stop menu navigation."""
        if self._active:
            run = state.get("run") or {}
            logger.info(
                "New run started (floor %s) - resuming agent",
                run.get("floor", "?"),
            )
            from sts2_agent.rewards import get_rewards_flow

            get_rewards_flow().clear()
        self.reset()

    def decide(self, state: dict) -> tuple[dict | None, list[str]]:
        state_type = str(state.get("state_type") or "").lower()
        now = time.time()

        if now < self._wait_until:
            remaining = self._wait_until - now
            return None, [f"menu flow pause ({remaining:.1f}s remaining)"]

        if state_type == "game_over":
            return self._decide_game_over(state, now)

        if state_type == "menu":
            self._active = True
            return self._decide_menu(state)

        if self._active:
            return None, ["menu flow active - waiting for menu/game_over state"]

        return None, ["menu flow idle"]

    def _decide_game_over(self, state: dict, now: float) -> tuple[dict | None, list[str]]:
        self._active = True

        if not self._logged_summary:
            reasons = log_run_summary(state)
            self._logged_summary = True
            if self._block_restart:
                return None, reasons + ["graceful shutdown - run finished, not restarting"]
            self._wait_until = now + POST_GAME_OVER_DELAY_SEC
            return None, reasons + [f"waiting {POST_GAME_OVER_DELAY_SEC}s before main menu"]

        if self._block_restart:
            return None, ["graceful shutdown - waiting for agent exit"]

        if not self._sent_main_menu:
            self._sent_main_menu = True
            self._wait_until = now + POST_MAIN_MENU_DELAY_SEC
            return {"action": "menu_select", "option": "main_menu"}, [
                "game over - menu_select main_menu",
                f"waiting {POST_MAIN_MENU_DELAY_SEC}s for menu to load",
            ]

        return {"action": "menu_select", "option": "main_menu"}, [
            "game over - retry main_menu"
        ]

    def _decide_menu(self, state: dict) -> tuple[dict | None, list[str]]:
        screen = get_menu_screen(state)
        options = extract_menu_options(state)
        opts_norm = {_normalize_option(o): o for o in options}
        reasons = [f"menu screen={screen or 'main'} options={options}"]

        if not options:
            return None, reasons + ["no menu options - wait"]

        # Abandon-run confirmation: abandon_run → popup yes/no → yes → singleplayer.
        if screen == "popup":
            yes_opt = _pick_option(options, "yes")
            no_opt = _pick_option(options, "no")
            if yes_opt and no_opt and (
                self._awaiting_abandon_confirm or self.abandon_stale_runs
            ):
                self._awaiting_abandon_confirm = False
                return {"action": "menu_select", "option": yes_opt}, reasons + [
                    "confirm abandon run - yes"
                ]

        if self._awaiting_abandon_confirm and screen in ("", "main"):
            return None, reasons + ["waiting for abandon confirm popup"]

        # Stale in-progress save (continue/abandon menu) - abandon to reach singleplayer.
        if (
            self.abandon_stale_runs
            and not self._awaiting_abandon_confirm
            and screen in ("", "main")
            and _pick_option(options, "abandon_run")
            and not _pick_option(options, "singleplayer")
        ):
            abandon = _pick_option(options, "abandon_run")
            self._awaiting_abandon_confirm = True
            self._wait_until = time.time() + POST_ABANDON_DELAY_SEC
            return {"action": "menu_select", "option": abandon}, reasons + [
                "abandon stale run - confirm with yes on next popup",
                f"waiting {POST_ABANDON_DELAY_SEC}s for popup",
            ]

        # Main menu
        if _pick_option(options, "singleplayer"):
            self._awaiting_abandon_confirm = False
            if screen in ("", "main") or "singleplayer" in opts_norm:
                return {"action": "menu_select", "option": "singleplayer"}, reasons + [
                    "start singleplayer"
                ]

        # Singleplayer mode picker
        if _pick_option(options, "standard"):
            if screen == "singleplayer" or "standard" in opts_norm:
                return {"action": "menu_select", "option": "standard"}, reasons + [
                    "singleplayer - standard run"
                ]

        # Tutorial prompt (skip)
        if _pick_option(options, "no"):
            if screen == "tutorial_prompt" or "no" in opts_norm:
                return {"action": "menu_select", "option": "no"}, reasons + [
                    "decline tutorial"
                ]

        # Character select - pick once, then confirm/embark (checkmark)
        if screen == "character_select":
            confirm = _pick_option(options, "confirm", "embark")
            char_opt = _pick_option(
                options,
                self.character,
                "ironclad",
                "the_ironclad",
            )
            selected = _selected_character(state)
            ready_to_start = self._character_picked or (
                selected is not None
                and _character_ids_match(selected, self.character)
            )

            if confirm and ready_to_start:
                self._character_picked = False
                return {"action": "menu_select", "option": confirm}, reasons + [
                    f"confirm character - start run ({confirm})"
                ]

            if char_opt and not ready_to_start:
                self._character_picked = True
                return {"action": "menu_select", "option": char_opt}, reasons + [
                    f"select character {char_opt}"
                ]

            if confirm:
                return {"action": "menu_select", "option": confirm}, reasons + [
                    f"confirm start ({confirm})"
                ]

            return None, reasons + ["character_select - waiting for confirm option"]

        # Confirm / embark (multiplayer load lobby)
        confirm = _pick_option(options, "confirm", "embark")
        if confirm and screen in ("multiplayer_load_lobby", ""):
            return {"action": "menu_select", "option": confirm}, reasons + [
                f"confirm start ({confirm})"
            ]

        # Popups - ignore/dismiss
        for dismiss in ("ignore", "back", "cancel"):
            opt = _pick_option(options, dismiss)
            if opt and screen == "popup":
                return {"action": "menu_select", "option": opt}, reasons + [
                    f"dismiss popup ({opt})"
                ]

        return None, reasons + ["unhandled menu screen - waiting"]
