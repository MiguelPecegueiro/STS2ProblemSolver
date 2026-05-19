"""Shared field extraction for dashboard DataFrames (handles pandas NaN)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return True
    return False


def coalesce_text(*values: object) -> str:
    for value in values:
        if is_missing(value):
            continue
        return str(value).strip()
    return ""


def reasoning_from_action_reasoning(text: object) -> str | None:
    """Free-text Qwen reasoning appended after qwen_macro tags in action_reasoning."""
    if is_missing(text):
        return None
    raw = str(text).strip()
    if "qwen_macro" not in raw.lower():
        return None
    parts = [p.strip() for p in raw.split(";")]
    if len(parts) < 3:
        return None
    candidate = parts[2]
    if not candidate or candidate.startswith("action="):
        return None
    if candidate.lower().startswith("qwen macro"):
        return None
    return candidate


def qwen_reasoning_from_record(
    qm: object,
    action_reasoning: object = None,
) -> str | None:
    """Best-effort Qwen reasoning from qwen_macro blob and/or action_reasoning."""
    if isinstance(qm, dict):
        parsed = qm.get("parsed")
        if isinstance(parsed, dict):
            from_parsed = parsed.get("reasoning")
            if not is_missing(from_parsed):
                return str(from_parsed).strip()

        response = qm.get("response")
        if isinstance(response, str) and response.strip():
            try:
                obj = json.loads(response.strip())
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                from_resp = obj.get("reasoning")
                if not is_missing(from_resp):
                    return str(from_resp).strip()

    from_action = reasoning_from_action_reasoning(action_reasoning)
    if from_action:
        return from_action
    return None
