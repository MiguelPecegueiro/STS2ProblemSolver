"""Structured decision logging to logs/run.log."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"
RUN_LOG_PATH = LOG_DIR / "run.log"

_file_handler: logging.FileHandler | None = None


def setup_decision_logging() -> None:
    """Attach a file handler for decision reasoning (idempotent)."""
    global _file_handler
    if _file_handler is not None:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(RUN_LOG_PATH, encoding="utf-8")
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )

    decision_logger = logging.getLogger("sts2_agent.decisions")
    decision_logger.setLevel(logging.INFO)
    decision_logger.addHandler(_file_handler)
    decision_logger.propagate = False


def log_decision(
    state_type: str,
    action: dict | None,
    reasons: list[str],
    *,
    extra: dict | None = None,
) -> None:
    setup_decision_logging()
    logger = logging.getLogger("sts2_agent.decisions")
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    lines = [
        f"--- decision [{ts}] state={state_type} ---",
        f"action: {action}",
    ]
    for reason in reasons:
        lines.append(f"  - {reason}")
    if extra:
        for key, val in extra.items():
            lines.append(f"  [{key}] {val}")
    logger.info("\n".join(lines))
