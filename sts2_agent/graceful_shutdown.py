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

        state_type = str(state.get("state_type") or "").lower()

        # Requested while idle between runs - nothing to finish.
        if not self._was_in_run and not run_in_progress:
            return True

        # Run ended: pipeline cleared after game_over / return to menu.
        from sts2_agent.data_pipeline import get_pipeline

        pipeline = get_pipeline()
        run_still_active = bool(getattr(pipeline, "_run_active", False))

        if self._saw_game_over and not run_still_active:
            return True

        if (
            self._was_in_run
            and not run_in_progress
            and state_type in ("menu", "game_over", "")
            and not run_still_active
        ):
            return True

        return False


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
