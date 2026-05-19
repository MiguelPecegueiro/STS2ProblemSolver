"""Run Explorer — deep dive on a single agent run (diagnostic)."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.fields import coalesce_text, is_missing, qwen_reasoning_from_record
from dashboard.metrics import (
    format_enemy_label,
    resolve_death_category,
    resolve_death_enemy,
    run_has_combat_summary,
)

PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 360
DEFAULT_AGENT_VERSION = "ppo_v6"
FILTER_EXPLORER_VERSION = "dash_explorer_agent_version"
FILTER_EXPLORER_RUN = "dash_explorer_run_id"

COMBAT_TYPES = frozenset({"monster", "elite", "boss", "hand_select"})
# card_select during combat should not split fights (matches data_pipeline fix)
COMBAT_SEGMENT_TYPES = COMBAT_TYPES | {"card_select"}
MAP_OPTION_RE = re.compile(
    r"option\[(\d+)\]\s*([^:\n]+?):\s*(\d+)",
    re.IGNORECASE,
)
POLICY_MAP_IDX_RE = re.compile(r"choose_map_node:(\d+)", re.IGNORECASE)
SOLVER_TAG_RE = re.compile(
    r"solver:\s*(lethal T1|setup T2 lethal|aggressive \(scales\)|trade|executing cached plan)",
    re.IGNORECASE,
)
CHOSEN_MAP_RE = re.compile(
    r"(?:choose|pick|select|chose)\s+.*?option\[(\d+)\]|index\s+(\d+)",
    re.IGNORECASE,
)


def _empty_figure(title: str, message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": "#888"},
    )
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def explorer_candidate_runs(
    runs: pd.DataFrame,
    *,
    agent_version: str,
) -> pd.DataFrame:
    """Agent runs for one version with Phase B combat_summary, best floor first."""
    if runs.empty:
        return runs.iloc[0:0]
    out = runs.copy()
    if "source" in out.columns:
        out = out[out["source"] != "human"]
    if "agent_version" in out.columns:
        out = out[out["agent_version"].astype(str) == str(agent_version)]
    if "combat_summary" in out.columns:
        out = out[out["combat_summary"].apply(run_has_combat_summary)]
    if out.empty:
        return out
    sort_cols = [c for c in ("floors_reached", "run_score", "timestamp") if c in out.columns]
    if sort_cols:
        ascending = [False, False, False][: len(sort_cols)]
        out = out.sort_values(sort_cols, ascending=ascending)
    return out.reset_index(drop=True)


def _run_label(row: pd.Series) -> str:
    floor = int(row.get("floors_reached") or 0)
    won = "WIN" if row.get("won") else "LOSS"
    rid = str(row.get("run_id") or "")[:8]
    ts = row.get("timestamp")
    ts_s = ""
    if pd.notna(ts):
        try:
            ts_s = pd.Timestamp(ts).strftime("%m-%d %H:%M")
        except (TypeError, ValueError):
            ts_s = str(ts)[:16]
    score = int(row.get("run_score") or 0)
    return f"F{floor} · {won} · score {score} · {rid} · {ts_s}"


def _hp_at_death(run: dict[str, Any]) -> int | None:
    after = run.get("hp_after_each_combat") or []
    if isinstance(after, list) and after:
        return int(after[-1])
    fights = run.get("combat_summary") or []
    if isinstance(fights, list) and fights:
        last = fights[-1]
        if isinstance(last, dict) and last.get("hp_end") is not None:
            return int(last["hp_end"])
    return None


MACRO_SCREEN_TYPES = frozenset(
    {
        "map",
        "card_reward",
        "rewards",
        "treasure",
        "rest_site",
        "shop",
        "fake_merchant",
        "event",
    }
)


def _qwen_macro_dict(row: pd.Series) -> dict[str, Any] | None:
    qm = row.get("qwen_macro")
    if isinstance(qm, dict):
        return qm
    if isinstance(qm, str):
        try:
            parsed = json.loads(qm)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _macro_choice_summary(row: pd.Series) -> str:
    action = str(row.get("action") or "")
    if action == "select_card_reward":
        picked = row.get("card_reward_picked")
        if picked:
            return str(picked)
        if pd.notna(row.get("card_index")):
            return f"card #{int(row['card_index'])}"
    if action == "skip_card_reward":
        return "(skip)"
    if action == "choose_map_node":
        room = str(row.get("map_room_chosen") or "").strip()
        idx = row.get("map_choice_index") if pd.notna(row.get("map_choice_index")) else row.get("action_index")
        if room:
            return f"{room} [{idx}]" if idx is not None and pd.notna(idx) else room
        if idx is not None and pd.notna(idx):
            return f"node [{int(idx)}]"
    if action == "choose_rest_option" and pd.notna(row.get("action_index")):
        return f"rest option [{int(row['action_index'])}]"
    if action == "shop_purchase" and pd.notna(row.get("action_index")):
        return f"buy [{int(row['action_index'])}]"
    if action == "choose_event_option" and pd.notna(row.get("action_index")):
        return f"event option [{int(row['action_index'])}]"
    if action in ("proceed", "advance_dialogue", "claim_reward", "claim_treasure_relic"):
        return action.replace("_", " ")
    if action:
        return action
    return "—"


def _full_reasoning_text(text: object) -> str | None:
    if is_missing(text):
        return None
    s = str(text).strip()
    if not s or s == "—" or s.lower() == "nan":
        return None
    return s


def _render_table_with_qwen_reasoning(
    df: pd.DataFrame,
    *,
    compact_cols: list[str] | None = None,
    reasoning_col: str = "qwen_reasoning_full",
    fallback_reasoning_col: str = "qwen_reasoning",
    expander_label: str = "Qwen reasoning (full text)",
    expanded: bool = True,
    row_title: Callable[[pd.Series], str] | None = None,
) -> None:
    """Compact dataframe plus wrapped full reasoning (avoids clipped table cells)."""
    if df.empty:
        return

    table = df.copy()
    reason_key = reasoning_col if reasoning_col in table.columns else fallback_reasoning_col
    reasons: list[tuple[str, str]] = []
    if reason_key in table.columns:
        for _, row in table.iterrows():
            full = _full_reasoning_text(row.get(reason_key))
            if not full:
                continue
            if row_title is not None:
                label = row_title(row)
            else:
                label = f"Floor {row.get('floor', '?')}"
            reasons.append((label, full))
        drop_cols = [c for c in (reasoning_col, fallback_reasoning_col) if c in table.columns]
        table = table.drop(columns=drop_cols, errors="ignore")

    if compact_cols:
        table = table[[c for c in compact_cols if c in table.columns]]

    st.dataframe(table, use_container_width=True, hide_index=True)

    if not reasons:
        return
    with st.expander(expander_label, expanded=expanded):
        for label, full in reasons:
            st.markdown(f"**{label}**")
            st.markdown(full)
            st.divider()


def macro_qwen_log_for_run(decisions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Macro-screen decisions with Qwen prompt/response metadata when present."""
    if decisions.empty or "run_id" not in decisions.columns:
        return pd.DataFrame()
    sub = decisions[decisions["run_id"].astype(str) == str(run_id)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["_st"] = sub["state_type"].astype(str).str.lower()
    sub = sub[sub["_st"].isin(MACRO_SCREEN_TYPES)]
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        qm = _qwen_macro_dict(row)
        reasoning = coalesce_text(
            qwen_reasoning_from_record(qm, row.get("action_reasoning")),
            row.get("qwen_macro_reasoning"),
        )
        source = row.get("qwen_macro_source") or (qm.get("source") if qm else None)
        if not source:
            source = "qwen" if qm and qm.get("user_prompt") else "rules"
        rows.append(
            {
                "floor": row.get("floor"),
                "act": row.get("act"),
                "screen": row["_st"],
                "action": row.get("action"),
                "choice": _macro_choice_summary(row),
                "source": source,
                "qwen_reasoning_full": reasoning or None,
                "has_qwen": bool(qm and qm.get("user_prompt")),
                "prompt": qm.get("user_prompt") if qm else None,
                "response": qm.get("response") if qm else None,
                "system_prompt": qm.get("system_prompt") if qm else None,
                "qwen_error": qm.get("error") or qm.get("validation_error") if qm else None,
                "timestamp": row.get("timestamp"),
            }
        )
    out = pd.DataFrame(rows)
    if "timestamp" in out.columns:
        out = out.sort_values("timestamp", na_position="last")
    return out.reset_index(drop=True)


def _qwen_summary(strategy: object) -> str:
    if not isinstance(strategy, dict):
        return "—"
    parts = [
        str(strategy.get("strategy") or "").strip(),
        f"dmg×{strategy.get('damage_multiplier', '?')}",
        f"hp×{strategy.get('hp_loss_multiplier', '?')}",
    ]
    reasoning = str(strategy.get("reasoning") or "").strip()
    text = " · ".join(p for p in parts if p and p != "·")
    if reasoning:
        text = f"{text} — {reasoning}" if text else reasoning
    return text or "—"


def _solver_tags_from_text(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    for m in SOLVER_TAG_RE.finditer(text):
        tag = m.group(1).strip()
        if tag.lower() == "executing cached plan":
            tag = "cached plan"
        found.append(tag)
    # de-dupe preserve order
    seen: set[str] = set()
    out: list[str] = []
    for t in found:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _split_combat_decisions(decisions: pd.DataFrame) -> list[pd.DataFrame]:
    """Split ordered decisions into combat segments (one per fight)."""
    if decisions.empty:
        return []
    segments: list[pd.DataFrame] = []
    current: list[dict] = []
    for _, row in decisions.iterrows():
        st = str(row.get("state_type") or "").lower()
        if st in COMBAT_SEGMENT_TYPES:
            current.append(row.to_dict())
        else:
            if current:
                segments.append(pd.DataFrame(current))
                current = []
    if current:
        segments.append(pd.DataFrame(current))
    return segments


def _enemy_label_from_segment(seg: pd.DataFrame) -> str:
    names: set[str] = set()
    if "combat_enemies" in seg.columns:
        for cell in seg["combat_enemies"]:
            if isinstance(cell, list):
                for n in cell:
                    if n:
                        names.add(str(n))
    return format_enemy_label(sorted(names)) if names else "Unknown"


def _encounter_label(fight: dict[str, Any]) -> str:
    return format_enemy_label(fight.get("enemy_names"))


def _should_merge_fights(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Same encounter split by card_select / logging (e.g. 3× Bygone Effigy)."""
    if a.get("state_type") != b.get("state_type"):
        return False
    la = _encounter_label(a)
    lb = _encounter_label(b)
    if la == "Unknown" or lb == "Unknown":
        return False
    return la == lb


def _combine_fight_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = group[0], group[-1]
    qwen = None
    for g in group:
        if g.get("qwen_strategy"):
            qwen = g["qwen_strategy"]
    combined = {
        "enemy_names": first.get("enemy_names") or last.get("enemy_names"),
        "turns": sum(int(g.get("turns") or 0) for g in group),
        "damage_taken": sum(int(g.get("damage_taken") or 0) for g in group),
        "damage_dealt": sum(int(g.get("damage_dealt") or 0) for g in group),
        "hp_start": first.get("hp_start"),
        "hp_end": last.get("hp_end"),
        "won_fight": last.get("won_fight"),
        "state_type": first.get("state_type"),
        "qwen_strategy": qwen,
        "_merged_segments": len(group),
    }
    return combined


def prepare_run_combat_view(
    run: dict[str, Any],
    decisions: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[int | None], list[int | None], list[str]]:
    """
    Merge split combat_summary rows, align HP arrays, infer missing enemy names.

    Returns (fights, hp_before, hp_after, segment_labels).
    """
    summary = [f for f in (run.get("combat_summary") or []) if isinstance(f, dict)]
    hp_before: list[int | None] = list(run.get("hp_before_each_combat") or [])
    hp_after: list[int | None] = list(run.get("hp_after_each_combat") or [])

    while len(hp_before) < len(summary):
        hp_before.append(None)
    while len(hp_after) < len(summary):
        hp_after.append(None)

    segments = _split_combat_decisions(decisions)
    segment_labels = [_enemy_label_from_segment(seg) for seg in segments]

    merged: list[dict[str, Any]] = []
    merged_b: list[int | None] = []
    merged_a: list[int | None] = []
    i = 0
    while i < len(summary):
        group = [summary[i]]
        j = i + 1
        while j < len(summary) and _should_merge_fights(summary[i], summary[j]):
            group.append(summary[j])
            j += 1
        merged.append(_combine_fight_group(group))
        merged_b.append(
            int(group[0].get("hp_start"))
            if group[0].get("hp_start") is not None
            else (int(hp_before[i]) if hp_before[i] is not None else None)
        )
        merged_a.append(
            int(group[-1].get("hp_end"))
            if group[-1].get("hp_end") is not None
            else (int(hp_after[j - 1]) if hp_after[j - 1] is not None else None)
        )
        i = j

    for idx, fight in enumerate(merged):
        if _encounter_label(fight) != "Unknown":
            continue
        label = None
        if idx < len(segment_labels) and segment_labels[idx] != "Unknown":
            label = segment_labels[idx]
        else:
            for j in range(idx, len(segment_labels)):
                if segment_labels[j] != "Unknown":
                    label = segment_labels[j]
                    break
            if label is None:
                for j in range(len(segment_labels)):
                    if segment_labels[j] != "Unknown":
                        label = segment_labels[j]
                        break
        if label:
            fight["enemy_names"] = [n.strip() for n in label.split(",") if n.strip()]
            fight["_inferred_from_decisions"] = True

    return merged, merged_b, merged_a, segment_labels


def _solver_tags_per_fight(decisions: pd.DataFrame, fight_count: int) -> list[str]:
    segments = _split_combat_decisions(decisions)
    tags: list[str] = []
    for seg in segments:
        blob = " ".join(
            str(x) for x in seg.get("action_reasoning", pd.Series(dtype=str)).dropna()
        )
        found = _solver_tags_from_text(blob)
        tags.append(", ".join(found) if found else "—")
    # Align tag list to merged fight count (merged fights <= raw segments)
    if len(tags) > fight_count:
        collapsed: list[str] = []
        step = max(1, len(tags) // max(fight_count, 1))
        for i in range(0, len(tags), step):
            chunk = tags[i : i + step]
            merged_tags: list[str] = []
            for t in chunk:
                merged_tags.extend(x.strip() for x in t.split(",") if x.strip() and x != "—")
            collapsed.append(", ".join(dict.fromkeys(merged_tags)) if merged_tags else "—")
        tags = collapsed[:fight_count]
    while len(tags) < fight_count:
        tags.append("—")
    return tags[:fight_count]


def _parse_map_choices_from_reasoning(text: str) -> dict[int, str]:
    options: dict[int, str] = {}
    for m in MAP_OPTION_RE.finditer(text or ""):
        idx = int(m.group(1))
        room = m.group(2).strip()
        options[idx] = room
    return options


def _map_options_from_row(row: pd.Series) -> dict[int, str]:
    options: dict[int, str] = {}
    raw = row.get("map_choice_options")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                idx = item.get("index")
                room = str(item.get("room") or "").strip()
                if idx is not None:
                    options[int(idx)] = room
            elif isinstance(item, str):
                m = re.match(r"\[(\d+)\]\s*(.+)", item.strip())
                if m:
                    options[int(m.group(1))] = m.group(2).strip()
    if not options:
        options = _parse_map_choices_from_reasoning(str(row.get("action_reasoning") or ""))
    return options


def _chosen_map_index(row: pd.Series) -> int | None:
    if pd.notna(row.get("map_choice_index")):
        return int(row["map_choice_index"])
    if pd.notna(row.get("action_index")):
        return int(row["action_index"])
    reasoning = str(row.get("action_reasoning") or "")
    m = POLICY_MAP_IDX_RE.search(reasoning)
    if m:
        return int(m.group(1))
    m = CHOSEN_MAP_RE.search(reasoning)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def map_path_for_run(decisions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Floor-by-floor map choices from map-screen decisions."""
    if decisions.empty or "run_id" not in decisions.columns:
        return pd.DataFrame()
    sub = decisions[decisions["run_id"].astype(str) == str(run_id)].copy()
    sub = sub[sub["state_type"].astype(str).str.lower() == "map"]
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, row in sub.iterrows():
        options = _map_options_from_row(row)
        chosen_idx = _chosen_map_index(row)
        room = str(row.get("map_room_chosen") or "").strip()
        if not room and chosen_idx is not None:
            room = options.get(chosen_idx, "")
        if not room:
            room = "?"
        source = "logged" if options else "policy only"
        qm = _qwen_macro_dict(row)
        qwen_reason = qwen_reasoning_from_record(qm, row.get("action_reasoning"))
        if is_missing(qwen_reason):
            qwen_reason = row.get("qwen_macro_reasoning")
        rows.append(
            {
                "floor": int(row["floor"]) if pd.notna(row.get("floor")) else None,
                "act": int(row["act"]) if pd.notna(row.get("act")) else None,
                "chosen_index": chosen_idx,
                "room": room,
                "options": ", ".join(f"[{i}] {r}" for i, r in sorted(options.items())),
                "source": source,
                "qwen_source": row.get("qwen_macro_source")
                or (qm.get("source") if isinstance(qm, dict) else None),
                "qwen_reasoning_full": qwen_reason or None,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "floor" in df.columns:
        # One row per floor — keep last map decision (drops duplicate polls on same floor)
        df = df.sort_values(["floor", "act"], na_position="last")
        df = df.drop_duplicates(subset=["floor"], keep="last")
    return df.reset_index(drop=True)


def card_picks_for_run(decisions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()
    sub = decisions[decisions["run_id"].astype(str) == str(run_id)].copy()
    mask = sub["state_type"].astype(str).str.lower() == "card_reward"
    if "card_reward_offered" in sub.columns:
        mask = mask | sub["card_reward_offered"].notna()
    sub = sub[mask]
    if sub.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, row in sub.iterrows():
        offered = row.get("card_reward_offered")
        if isinstance(offered, str):
            try:
                offered = json.loads(offered)
            except json.JSONDecodeError:
                offered = [offered]
        if not isinstance(offered, list):
            offered = offered if offered is None else [offered]
        picked = row.get("card_reward_picked")
        if not picked and str(row.get("action")) == "select_card_reward":
            picked = f"slot {row.get('card_index')}"
        qm = _qwen_macro_dict(row)
        qwen_reason = qwen_reasoning_from_record(qm, row.get("action_reasoning"))
        if is_missing(qwen_reason):
            qwen_reason = row.get("qwen_macro_reasoning")
        rows.append(
            {
                "floor": row.get("floor"),
                "act": row.get("act"),
                "offered": ", ".join(str(c) for c in (offered or [])),
                "picked": picked or ("(skip)" if str(row.get("action")) == "skip_card_reward" else "—"),
                "action": row.get("action"),
                "source": row.get("qwen_macro_source")
                or (qm.get("source") if isinstance(qm, dict) else "rules"),
                "qwen_reasoning_full": qwen_reason or None,
            }
        )
    return pd.DataFrame(rows)


def _timeline_moments(
    run: dict[str, Any],
    decisions: pd.DataFrame,
) -> list[dict[str, str]]:
    """Heuristic 'what went wrong' moments for diagnostics."""
    moments: list[dict[str, str]] = []
    run_id = str(run.get("run_id") or "")

    if not run.get("won"):
        enemy = resolve_death_enemy(run) or "unknown"
        cat = resolve_death_category(run)
        hp = _hp_at_death(run)
        moments.append(
            {
                "severity": "critical",
                "when": "End of run",
                "title": f"Death — {cat}",
                "detail": f"Killed by {enemy}" + (f" at {hp} HP" if hp is not None else ""),
            }
        )

    fights_merged, _, _, _ = prepare_run_combat_view(run, decisions)

    for i, fight in enumerate(fights_merged, start=1):
        enemies = _encounter_label(fight)
        dmg_taken = int(fight.get("damage_taken") or 0)
        hp_start = int(fight.get("hp_start") or 0)
        hp_end = int(fight.get("hp_end") or 0)
        won = fight.get("won_fight")

        if won is False:
            moments.append(
                {
                    "severity": "critical",
                    "when": f"Fight {i}",
                    "title": f"Lost fight vs {enemies}",
                    "detail": f"Took {dmg_taken} damage ({hp_start}→{hp_end} HP)",
                }
            )
        elif hp_start > 0 and hp_end <= int(hp_start * 0.35):
            moments.append(
                {
                    "severity": "warning",
                    "when": f"Fight {i}",
                    "title": f"Low HP after {enemies}",
                    "detail": f"Ended fight at {hp_end}/{hp_start} HP ({dmg_taken} damage taken)",
                }
            )
        elif dmg_taken >= 25:
            moments.append(
                {
                    "severity": "warning",
                    "when": f"Fight {i}",
                    "title": f"Heavy damage vs {enemies}",
                    "detail": f"{dmg_taken} damage taken ({hp_start}→{hp_end} HP)",
                }
            )

    if not decisions.empty and run_id:
        sub = decisions[decisions["run_id"].astype(str) == run_id]
        if "hp_lost_this_turn" in sub.columns:
            bad = sub[sub["hp_lost_this_turn"].fillna(0) >= 15]
            for _, row in bad.head(5).iterrows():
                moments.append(
                    {
                        "severity": "warning",
                        "when": f"Floor {row.get('floor', '?')}",
                        "title": "Spike damage on a turn",
                        "detail": (
                            f"{int(row['hp_lost_this_turn'])} HP lost — "
                            f"{str(row.get('state_type') or '')} — "
                            f"{str(row.get('action_reasoning') or '')[:100]}"
                        ),
                    }
                )

        for _, row in sub.iterrows():
            reason = str(row.get("action_reasoning") or "")
            if "rules fallback" in reason.lower() and "policy_net" in reason.lower():
                moments.append(
                    {
                        "severity": "info",
                        "when": f"Floor {row.get('floor', '?')}",
                        "title": "Policy rejected — rules fallback",
                        "detail": reason[:160],
                    }
                )
                break

    order = {"critical": 0, "warning": 1, "info": 2}
    moments.sort(key=lambda m: order.get(m["severity"], 9))
    return moments[:20]


def _hp_curve_figure(
    fights: list[dict[str, Any]],
    hp_before: list[int | None],
    hp_after: list[int | None],
) -> go.Figure:
    if not fights or not hp_before:
        return _empty_figure("HP across combats", "No combat HP data on this run")

    n = min(len(fights), len(hp_before), len(hp_after))
    labels: list[str] = []
    for i in range(n):
        label = _encounter_label(fights[i])
        if fights[i].get("_merged_segments", 1) > 1:
            label += f" (×{fights[i]['_merged_segments']})"
        if fights[i].get("_inferred_from_decisions"):
            label += " *"
        labels.append(label or f"Fight {i + 1}")

    x_start, y_start, x_end, y_end = [], [], [], []
    for i in range(n):
        if hp_before[i] is not None:
            x_start.append(i + 1)
            y_start.append(int(hp_before[i]))
        if hp_after[i] is not None:
            x_end.append(i + 1)
            y_end.append(int(hp_after[i]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_start,
            y=y_start,
            mode="lines+markers",
            name="HP start",
            line={"color": "#5dade2"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_end,
            y=y_end,
            mode="lines+markers",
            name="HP end",
            line={"color": "#e74c3c"},
        )
    )
    fig.update_layout(
        title="HP curve (per combat)",
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        xaxis_title="Combat #",
        yaxis_title="HP",
        legend={"orientation": "h", "y": 1.1},
    )
    fig.update_xaxes(tickvals=list(range(1, n + 1)), ticktext=labels, tickangle=-25)
    return fig


def _fights_table(
    fights: list[dict[str, Any]],
    solver_tags: list[str],
) -> pd.DataFrame:
    if not fights:
        return pd.DataFrame()
    rows: list[dict] = []
    for i, fight in enumerate(fights, start=1):
        enemies = _encounter_label(fight)
        note = ""
        if fight.get("_merged_segments", 1) > 1:
            note = f"merged ×{fight['_merged_segments']}"
        if fight.get("_inferred_from_decisions"):
            note = (note + "; " if note else "") + "names from decisions"
        rows.append(
            {
                "#": i,
                "type": fight.get("state_type") or "monster",
                "enemies": enemies,
                "notes": note or "—",
                "turns": fight.get("turns"),
                "damage dealt": fight.get("damage_dealt"),
                "damage taken": fight.get("damage_taken"),
                "HP": f"{fight.get('hp_start')}→{fight.get('hp_end')}",
                "won": "✓" if fight.get("won_fight") else "✗",
                "Qwen": _qwen_summary(fight.get("qwen_strategy")),
                "Solver tags": solver_tags[i - 1] if i - 1 < len(solver_tags) else "—",
            }
        )
    return pd.DataFrame(rows)


def render_run_explorer(runs: pd.DataFrame, decisions: pd.DataFrame) -> None:
    st.header("Run Explorer")
    st.caption(
        "Diagnose a single run end-to-end — fights, rewards, map, HP, and turning points. "
        "Default filter: **ppo_v6**."
    )

    versions = sorted(runs["agent_version"].dropna().astype(str).unique().tolist()) if (
        not runs.empty and "agent_version" in runs.columns
    ) else []
    if DEFAULT_AGENT_VERSION not in versions and versions:
        st.info(
            f"No **{DEFAULT_AGENT_VERSION}** runs in `data/runs.jsonl` yet. "
            f"Available: {', '.join(versions)}. Pick a version below."
        )

    version_options = versions or [DEFAULT_AGENT_VERSION]
    if FILTER_EXPLORER_VERSION not in st.session_state:
        st.session_state[FILTER_EXPLORER_VERSION] = (
            DEFAULT_AGENT_VERSION
            if DEFAULT_AGENT_VERSION in version_options
            else version_options[0]
        )
    if st.session_state[FILTER_EXPLORER_VERSION] not in version_options:
        st.session_state[FILTER_EXPLORER_VERSION] = version_options[0]

    col_v, col_run = st.columns([1, 2])
    with col_v:
        version = st.selectbox(
            "Agent version",
            version_options,
            key=FILTER_EXPLORER_VERSION,
        )
    candidates = explorer_candidate_runs(runs, agent_version=version)
    with col_run:
        if candidates.empty:
            st.warning(f"No Phase B runs for **{version}** (need non-empty `combat_summary`).")
            return
        labels = [_run_label(candidates.iloc[i]) for i in range(len(candidates))]
        if FILTER_EXPLORER_RUN not in st.session_state:
            st.session_state[FILTER_EXPLORER_RUN] = str(candidates.iloc[0].get("run_id"))
        run_ids = candidates["run_id"].astype(str).tolist()
        if st.session_state[FILTER_EXPLORER_RUN] not in run_ids:
            st.session_state[FILTER_EXPLORER_RUN] = run_ids[0]
        pick = st.selectbox(
            "Run (best floor first)",
            run_ids,
            format_func=lambda rid: labels[run_ids.index(rid)],
            key=FILTER_EXPLORER_RUN,
        )

    run_row = candidates[candidates["run_id"].astype(str) == str(pick)].iloc[0]
    run = run_row.to_dict()
    run_dec = (
        decisions[decisions["run_id"].astype(str) == str(pick)].copy()
        if not decisions.empty and "run_id" in decisions.columns
        else pd.DataFrame()
    )
    if "timestamp" in run_dec.columns:
        run_dec = run_dec.sort_values("timestamp")

    st.subheader("Run summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Floor reached", int(run.get("floors_reached") or 0))
    c2.metric("Run score", int(run.get("run_score") or 0))
    hp_death = _hp_at_death(run)
    c3.metric("HP at death", hp_death if hp_death is not None else "—")
    c4.metric("Outcome", "Win" if run.get("won") else "Loss")

    c5, c6, c7 = st.columns(3)
    c5.write(f"**Death category:** {resolve_death_category(run)}")
    enemy = resolve_death_enemy(run)
    c6.write(f"**Killing enemy:** {enemy or '—'}")
    cause = run.get("cause_of_death")
    c7.write(f"**Cause:** {cause if cause else '—'}")

    fights_merged, hp_b, hp_a, _ = prepare_run_combat_view(run, run_dec)
    solver_tags = _solver_tags_per_fight(run_dec, len(fights_merged))

    st.subheader("Fight breakdown")
    st.caption(
        "Consecutive rows for the same enemy (e.g. Bygone Effigy) are merged — "
        "that was one fight interrupted by card_select overlays. "
        "`*` = enemy names inferred from combat decisions."
    )
    fights_df = _fights_table(fights_merged, solver_tags)
    if fights_df.empty:
        st.caption("No `combat_summary` on this run.")
    else:
        st.dataframe(fights_df, use_container_width=True, hide_index=True)

    st.subheader("Card picks")
    cards_df = card_picks_for_run(run_dec, str(pick))
    if cards_df.empty:
        st.caption("No card_reward decisions logged for this run.")
    else:
        _render_table_with_qwen_reasoning(
            cards_df,
            compact_cols=["floor", "act", "offered", "picked", "action", "source"],
            expander_label="Qwen reasoning — card picks",
            expanded=True,
            row_title=lambda r: (
                f"Floor {r.get('floor', '?')} · picked {r.get('picked', '—')}"
            ),
        )

    st.subheader("Macro decisions (Qwen)")
    macro_df = macro_qwen_log_for_run(run_dec, str(pick))
    if macro_df.empty:
        st.caption(
            "No map / reward / shop / rest / event decisions logged for this run. "
            "New runs with macro Qwen enabled store `qwen_macro` prompts in `decisions.jsonl`."
        )
    else:
        _render_table_with_qwen_reasoning(
            macro_df,
            compact_cols=["floor", "act", "screen", "choice", "action", "source"],
            expander_label="Qwen reasoning — macro decisions",
            expanded=False,
            row_title=lambda r: (
                f"Floor {r.get('floor', '?')} · {r.get('screen', '?')} · {r.get('choice', '—')}"
            ),
        )
        qwen_rows = macro_df[macro_df["has_qwen"] == True]  # noqa: E712
        if qwen_rows.empty:
            st.caption("No Qwen API traces on this run (rules-only macro or pre-logging build).")
        else:
            with st.expander(f"Qwen prompts & responses ({len(qwen_rows)} decisions)", expanded=False):
                for i, (_, mrow) in enumerate(qwen_rows.iterrows()):
                    label = (
                        f"Floor {mrow.get('floor', '?')} · "
                        f"{mrow.get('screen', '?')} · {mrow.get('choice', '—')}"
                    )
                    st.markdown(f"**{label}** — source `{mrow.get('source', '?')}`")
                    if mrow.get("qwen_error"):
                        st.warning(str(mrow["qwen_error"]))
                    full_reason = _full_reasoning_text(mrow.get("qwen_reasoning_full"))
                    if full_reason:
                        st.markdown("**Reasoning**")
                        st.markdown(full_reason)
                    if mrow.get("system_prompt"):
                        st.text("System prompt")
                        st.code(str(mrow["system_prompt"]), language=None)
                    if mrow.get("prompt"):
                        st.text("User prompt (Qwen input)")
                        st.text_area(
                            "user_prompt",
                            value=str(mrow["prompt"]),
                            height=min(400, 120 + len(str(mrow["prompt"])) // 4),
                            key=f"macro_qwen_prompt_{pick}_{i}",
                            label_visibility="collapsed",
                        )
                    if mrow.get("response"):
                        st.text("Model response")
                        st.code(str(mrow["response"]), language="json")
                    st.divider()

    st.subheader("Map path")
    map_df = map_path_for_run(run_dec, str(pick))
    if map_df.empty:
        st.caption("No map decisions logged for this run.")
    else:
        st.caption(
            "Older runs only log policy index (`choose_map_node:0`) — room names need "
            "rules map reasoning or new `map_choice_options` logging (fixed going forward)."
        )
        _render_table_with_qwen_reasoning(
            map_df,
            compact_cols=[
                "floor",
                "act",
                "chosen_index",
                "room",
                "options",
                "source",
                "qwen_source",
            ],
            expander_label="Qwen reasoning — map path",
            expanded=True,
            row_title=lambda r: (
                f"Floor {r.get('floor', '?')} · {r.get('room', '?')} "
                f"[{r.get('chosen_index', '?')}]"
            ),
        )

    st.subheader("HP curve")
    st.plotly_chart(_hp_curve_figure(fights_merged, hp_b, hp_a), use_container_width=True)

    st.subheader("Decision timeline")
    moments = _timeline_moments(run, run_dec)
    if not moments:
        st.caption("No notable issues flagged for this run.")
    else:
        for m in moments:
            icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}.get(m["severity"], "⚪")
            st.markdown(f"{icon} **{m['when']} — {m['title']}**  \n{m['detail']}")

    with st.expander("Raw run JSON"):
        st.json(
            {
                k: run.get(k)
                for k in (
                    "run_id",
                    "agent_version",
                    "floors_reached",
                    "run_score",
                    "won",
                    "cause_of_death",
                    "killing_enemy",
                    "combat_summary",
                    "hp_before_each_combat",
                    "hp_after_each_combat",
                )
            }
        )
