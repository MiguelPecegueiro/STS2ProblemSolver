"""Finish the active run before exiting (separate from immediate Ctrl+C)."""

from __future__ import annotations

import logging
import signal
from typing import Callable

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """When requested, keep playing until the current run ends, then stop."""

    def __init__(self) -> None:
        self.requested = False
        self._was_in_run = False
        self._saw_game_over = False

    def request(self) -> None:
        if self.requested:
            return
        self.requested = True
        logger.info(
            "Graceful shutdown requested - will stop after this run ends "
            "(no new run will be started)"
        )

    def observe(self, state: dict, *, run_in_progress: bool) -> None:
        if run_in_progress:
            self._was_in_run = True
        state_type = str(state.get("state_type") or "").lower()
        if state_type == "game_over":
            self._saw_game_over = True

    def should_exit(self, state: dict, *, run_in_progress: bool) -> bool:
        if not self.requested:
            return False

        # Requested while idle between runs - nothing to finish.
        if not self._was_in_run and not run_in_progress:
            return True

        # Run ended: pipeline cleared after game_over / return to menu.
        from sts2_agent.data_pipeline import get_pipeline

        pipeline = get_pipeline()
        run_still_active = bool(getattr(pipeline, "_run_active", False))

        return _run_finished_for_exit(
            state,
            run_in_progress=run_in_progress,
            saw_game_over=self._saw_game_over,
            require_was_in_run=True,
            was_in_run=self._was_in_run,
        )


def install_graceful_shutdown_handler(on_request: Callable[[], None]) -> None:
    """Register OS-specific 'finish run then stop' signal (not Ctrl+C)."""

    def _handler(signum: int, frame: object) -> None:
        on_request()

    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _handler)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handler)


def shutdown_help_message() -> str:
    parts = ["Ctrl+C to stop immediately"]
    if hasattr(signal, "SIGBREAK"):
        parts.append("Ctrl+Break to finish the current run, then stop")
    if hasattr(signal, "SIGUSR1"):
        parts.append("SIGUSR1 (kill -USR1 <pid>) to finish the current run, then stop")
    return " | ".join(parts)


class SingleRunController:
    """Exit after one completed run (no menu restart for a second run)."""

    def __init__(self) -> None:
        self._was_in_run = False
        self._saw_game_over = False
        self._blocked_restart = False

    def observe(self, state: dict, *, run_in_progress: bool) -> None:
        if run_in_progress:
            self._was_in_run = True
        state_type = str(state.get("state_type") or "").lower()
        if state_type == "game_over":
            self._saw_game_over = True

    def apply_menu_block(self, menu_flow: object) -> None:
        if self._saw_game_over and not self._blocked_restart:
            block = getattr(menu_flow, "block_restart", None)
            if callable(block):
                block()
            self._blocked_restart = True

    def should_exit(
        self,
        state: dict,
        *,
        run_in_progress: bool,
        menu_flow: object,
    ) -> bool:
        if not self._was_in_run:
            return False
        self.apply_menu_block(menu_flow)
        return _run_finished_for_exit(
            state,
            run_in_progress=run_in_progress,
            saw_game_over=self._saw_game_over,
        )


def _run_finished_for_exit(
    state: dict,
    *,
    run_in_progress: bool,
    saw_game_over: bool,
    require_was_in_run: bool = False,
    was_in_run: bool = True,
) -> bool:
    if require_was_in_run and not was_in_run:
        return False

    from sts2_agent.data_pipeline import get_pipeline

    state_type = str(state.get("state_type") or "").lower()
    pipeline = get_pipeline()
    run_still_active = bool(getattr(pipeline, "_run_active", False))

    if saw_game_over and not run_still_active:
        return True

    if (
        was_in_run
        and not run_in_progress
        and state_type in ("menu", "game_over", "")
        and not run_still_active
    ):
        return True

    return False
