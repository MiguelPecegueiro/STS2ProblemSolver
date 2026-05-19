"""Dashboard field extraction helpers."""

from __future__ import annotations

import math

import pandas as pd

from dashboard.fields import is_missing, qwen_reasoning_from_record


def test_nan_is_missing() -> None:
    assert is_missing(float("nan"))
    assert is_missing(pd.NA)


def test_nan_or_chain_bug_fixed() -> None:
    qm = {
        "parsed": {"action": "choose_map_node", "index": 0, "reasoning": "Pick monster path"},
        "response": '{"action":"choose_map_node","index":0,"reasoning":"Pick monster path"}',
    }
    # Simulates broken `nan or parsed.reasoning` when column was NaN
    row_reasoning = float("nan")
    assert qwen_reasoning_from_record(qm, row_reasoning) == "Pick monster path"


def test_reasoning_from_action_reasoning_third_segment() -> None:
    qm = {"parsed": {"action": "choose_event_option", "index": 0}}
    text = (
        "qwen_macro: map; action={'action': 'choose_map_node', 'index': 0}; "
        "Both nodes are monsters — take the first."
    )
    assert qwen_reasoning_from_record(qm, text) == "Both nodes are monsters — take the first."


def test_reasoning_from_response_when_parsed_lacks_it() -> None:
    qm = {
        "parsed": {"action": "proceed"},
        "response": (
            '{"action":"proceed","reasoning":"Only option is to proceed."}'
        ),
    }
    assert qwen_reasoning_from_record(qm, None) == "Only option is to proceed."
