"""STS2 Agent - live performance dashboard (read-only)."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.characters import normalize_character_name  # noqa: E402
from dashboard.metrics import incoming_damage_from_snapshot  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
RUNS_PATH = DATA_DIR / "runs.jsonl"
DECISIONS_PATH = DATA_DIR / "decisions.jsonl"
CARD_CHOICES_PATH = DATA_DIR / "card_choices.jsonl"

PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 380

DATE_FILTER_OPTIONS = (
    "All time",
    "Last month",
    "Last 3 months",
    "Last 6 months",
    "Last year",
    "Custom range",
)
DATE_PRESET_DAYS = {
    "Last month": 30,
    "Last 3 months": 90,
    "Last 6 months": 180,
    "Last year": 365,
}

VIEW_HUMAN = "Human only"
VIEW_AGENT = "Agent only"
VIEW_COMPARE = "Compare (human vs agent)"
VIEW_COMPARE_VERSIONS = "Compare versions"
VIEW_MODES = (VIEW_HUMAN, VIEW_AGENT, VIEW_COMPARE, VIEW_COMPARE_VERSIONS)

DEFAULT_AGENT_VERSION = "rules_v1"
DEFAULT_DASHBOARD_CHARACTER = normalize_character_name("ironclad")

# Streamlit session_state keys - persist sidebar filters across "Reload data".
FILTER_VIEW_MODE = "dash_filter_view_mode"
FILTER_CHARACTER = "dash_filter_character"
FILTER_ASCENSION = "dash_filter_ascension"
FILTER_DATE_MODE = "dash_filter_date_mode"
FILTER_CUSTOM_DATES = "dash_filter_custom_dates"
FILTER_SAVED_AGENT_VERSIONS = "dash_saved_agent_versions"
FILTER_DECISION_RUN = "dash_filter_decision_run"
# Kept in sync with dashboard.sections.FILTER_DETAIL_VERSION
FILTER_DETAIL_VERSION = "dash_detail_agent_version"


def _ensure_select_value(key: str, options: list[str], default: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = default
        return
    if st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]


def _is_bc_agent_version(version: str) -> bool:
    """BC training tags (bc_v1, bc_v3, …) — hidden from default version selection."""
    v = str(version).strip().lower()
    return v == "bc" or v.startswith("bc_")


def _default_agent_versions(versions: list[str]) -> list[str]:
    """Default multiselect: all versions except BC (fallback to all if only BC exists)."""
    non_bc = [v for v in versions if not _is_bc_agent_version(v)]
    return non_bc if non_bc else list(versions)


def _saved_agent_versions(versions: list[str]) -> list[str]:
    legacy_key = "agent_versions_pick"
    if FILTER_SAVED_AGENT_VERSIONS not in st.session_state:
        if legacy_key in st.session_state:
            st.session_state[FILTER_SAVED_AGENT_VERSIONS] = _default_agent_versions(
                list(st.session_state[legacy_key])
            )
        else:
            st.session_state[FILTER_SAVED_AGENT_VERSIONS] = _default_agent_versions(versions)
    saved = [v for v in st.session_state[FILTER_SAVED_AGENT_VERSIONS] if v in versions]
    if saved:
        return saved
    if not versions:
        return []
    # User explicitly cleared selection — do not reset.
    if st.session_state[FILTER_SAVED_AGENT_VERSIONS] == []:
        return []
    return _default_agent_versions(versions)


def version_label(version: str) -> str:
    return str(version)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows


def _data_revision_key() -> str:
    """Cache-bust when JSONL files change on disk."""
    from sts2_agent.enemy_compendium import COMPENDIUM_PATH, OBSERVATIONS_PATH

    parts: list[str] = []
    for path in (RUNS_PATH, DECISIONS_PATH, CARD_CHOICES_PATH, COMPENDIUM_PATH, OBSERVATIONS_PATH):
        if path.exists():
            parts.append(f"{path.name}:{path.stat().st_mtime_ns}:{path.stat().st_size}")
        else:
            parts.append(f"{path.name}:missing")
    return "|".join(parts)


@st.cache_data(ttl=30)
def load_runs(_revision: str) -> pd.DataFrame:
    """Read data/runs.jsonl into a DataFrame."""
    _ = _revision
    rows = _read_jsonl(RUNS_PATH)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "source" not in df.columns:
        df["source"] = "agent"
    else:
        df["source"] = df["source"].fillna("agent").astype(str)
    if "agent_version" not in df.columns:
        df["agent_version"] = DEFAULT_AGENT_VERSION
    else:
        is_agent = df["source"] != "human"
        df.loc[is_agent, "agent_version"] = (
            df.loc[is_agent, "agent_version"].fillna(DEFAULT_AGENT_VERSION).astype(str)
        )
    if "character" in df.columns:
        df["character"] = df["character"].map(normalize_character_name)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if "won" in df.columns:
        df["won"] = df["won"].astype(bool)
    for col in ("floors_reached", "act_reached", "ascension", "total_decisions", "total_damage_taken", "total_damage_dealt", "gold_at_death"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in ("avg_hp_pct_after_combat", "best_combat_hp_pct", "worst_combat_hp_pct"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "run_duration_sec" in df.columns:
        df["run_duration_sec"] = pd.to_numeric(df["run_duration_sec"], errors="coerce")
    if "run_score" in df.columns:
        df["run_score"] = pd.to_numeric(df["run_score"], errors="coerce").fillna(0)
    if "final_deck" in df.columns:
        df["deck_size"] = df["final_deck"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df["deck_size"] = 0
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["run_number"] = range(1, len(df) + 1)
    return df


@st.cache_data(ttl=30)
def load_decisions(_revision: str) -> pd.DataFrame:
    """Read data/decisions.jsonl into a flattened DataFrame."""
    _ = _revision
    rows = _read_jsonl(DECISIONS_PATH)
    if not rows:
        return pd.DataFrame()
    flat: list[dict] = []
    for row in rows:
        snap = row.get("state_snapshot") or {}
        if not isinstance(snap, dict):
            snap = {}
        action = row.get("action_taken") or {}
        outcome = row.get("run_outcome") or {}
        immediate = row.get("immediate_reward")
        block_applied = None
        hp_lost_this_turn = None
        damage_dealt_turn = None
        if isinstance(immediate, dict):
            block_applied = immediate.get("block_applied")
            hp_lost_this_turn = immediate.get("hp_lost_this_turn")
            damage_dealt_turn = immediate.get("damage_dealt")
        flat.append(
            {
                "run_id": row.get("run_id"),
                "timestamp": row.get("timestamp"),
                "floor": row.get("floor"),
                "act": row.get("act"),
                "state_type": row.get("state_type"),
                "action": action.get("action") if isinstance(action, dict) else None,
                "card_index": action.get("card_index") if isinstance(action, dict) else None,
                "action_reasoning": row.get("action_reasoning"),
                "immediate_reward": immediate,
                "block_applied": block_applied,
                "hp_lost_this_turn": hp_lost_this_turn,
                "damage_dealt_turn": damage_dealt_turn,
                "run_won": outcome.get("won") if isinstance(outcome, dict) else None,
                "agent_version": row.get("agent_version"),
                "player_hp": snap.get("player_hp"),
                "player_max_hp": snap.get("player_max_hp"),
                "player_energy": snap.get("player_energy"),
                "incoming_damage": incoming_damage_from_snapshot(snap),
                "hand": snap.get("hand"),
                "card_reward_offered": row.get("card_reward_offered")
                or snap.get("card_reward_offered"),
                "card_reward_picked": row.get("card_reward_picked")
                or snap.get("card_reward_picked"),
            }
        )
    df = pd.DataFrame(flat)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ("floor", "act", "card_index", "block_applied", "hp_lost_this_turn", "damage_dealt_turn"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(ttl=30)
def load_card_choices(_revision: str) -> pd.DataFrame:
    """Human card reward picks from data/card_choices.jsonl."""
    _ = _revision
    rows = _read_jsonl(CARD_CHOICES_PATH)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns and "run_id" in df.columns:
        pass
    for col in ("floor", "act", "ascension", "hp_at_pick"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "won" in df.columns:
        df["won"] = df["won"].astype(bool)
    if "character" in df.columns:
        df["character"] = df["character"].map(normalize_character_name)
    return df


def split_by_source(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Human vs agent subsets (agent = anything not marked human)."""
    if runs.empty:
        return runs.iloc[0:0], runs.iloc[0:0]
    if "source" not in runs.columns:
        return runs.iloc[0:0], runs.copy()
    human = runs[runs["source"] == "human"]
    agent = runs[runs["source"] != "human"]
    return human, agent


def agent_runs_only(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    if "source" in runs.columns:
        return runs[runs["source"] != "human"]
    return runs


def available_agent_versions(runs: pd.DataFrame) -> list[str]:
    """Sorted agent_version tags present in the dataset."""
    agent = agent_runs_only(runs)
    if agent.empty or "agent_version" not in agent.columns:
        return []
    return sorted(agent["agent_version"].dropna().astype(str).unique().tolist())


def agent_version_groups(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split agent runs by agent_version (supports any number of versions)."""
    agent = agent_runs_only(runs)
    if agent.empty or "agent_version" not in agent.columns:
        return {}
    groups: dict[str, pd.DataFrame] = {}
    for version in available_agent_versions(runs):
        subset = agent[agent["agent_version"] == version]
        if not subset.empty:
            groups[str(version)] = subset
    return groups


def version_count_caption(runs: pd.DataFrame) -> str:
    groups = agent_version_groups(runs)
    if not groups:
        return ""
    parts = [f"{len(df)} {version_label(v)}" for v, df in groups.items()]
    return " · ".join(parts)


def renumber_runs(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    out = runs.sort_values("timestamp").reset_index(drop=True)
    out = out.copy()
    out["run_number"] = range(1, len(out) + 1)
    return out


def decisions_for_runs(decisions: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or runs.empty or "run_id" not in decisions.columns:
        return decisions.iloc[0:0]
    ids = set(runs["run_id"].astype(str))
    return decisions[decisions["run_id"].astype(str).isin(ids)]


def agent_decisions_by_version(
    decisions: pd.DataFrame,
    runs: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Split agent decisions by agent_version (row tag or join via run_id)."""
    if decisions.empty:
        return {}
    dec = decisions.copy()
    if "agent_version" not in dec.columns or dec["agent_version"].isna().all():
        if runs is None or runs.empty or "run_id" not in dec.columns:
            return {"Agent": dec}
        agent_runs = agent_runs_only(runs)
        if agent_runs.empty or "agent_version" not in agent_runs.columns:
            return {"Agent": dec}
        ver_by_run = agent_runs.set_index(agent_runs["run_id"].astype(str))["agent_version"].astype(str)
        dec["agent_version"] = dec["run_id"].astype(str).map(ver_by_run)
    dec["agent_version"] = dec["agent_version"].fillna(DEFAULT_AGENT_VERSION).astype(str)
    groups: dict[str, pd.DataFrame] = {}
    for ver in sorted(dec["agent_version"].dropna().unique()):
        subset = dec[dec["agent_version"] == ver]
        if not subset.empty:
            groups[str(ver)] = subset
    return groups


def render_by_agent_versions(runs: pd.DataFrame, render_fn) -> None:
    """Side-by-side columns for each agent_version present in the data."""
    groups = agent_version_groups(runs)
    if not groups:
        render_fn(renumber_runs(agent_runs_only(runs)), "Agent")
        return
    if len(groups) == 1:
        ver, subset = next(iter(groups.items()))
        render_fn(renumber_runs(subset), version_label(ver))
        return

    versions = list(groups.keys())
    cols = st.columns(len(versions))
    for col, ver in zip(cols, versions):
        with col:
            render_fn(renumber_runs(groups[ver]), version_label(ver))


def render_by_source(view_mode: str, runs: pd.DataFrame, render_fn) -> None:
    """Invoke render_fn(runs_subset, label) once or side-by-side for Compare."""
    if view_mode == VIEW_COMPARE_VERSIONS:
        render_by_agent_versions(runs, render_fn)
        return

    if view_mode == VIEW_COMPARE:
        human, agent = split_by_source(runs)
        col_h, col_a = st.columns(2)
        with col_h:
            if human.empty:
                st.caption("No human runs for these filters.")
            else:
                render_fn(renumber_runs(human), "Human")
        with col_a:
            if agent.empty:
                st.caption("No agent runs for these filters.")
            else:
                groups = agent_version_groups(agent)
                if len(groups) > 1:
                    render_by_agent_versions(agent, render_fn)
                elif len(groups) == 1:
                    ver = next(iter(groups))
                    render_fn(renumber_runs(groups[ver]), version_label(ver))
                else:
                    render_fn(renumber_runs(agent), "Agent")
        return

    if view_mode == VIEW_AGENT:
        groups = agent_version_groups(runs)
        if len(groups) > 1:
            render_by_agent_versions(runs, render_fn)
            return
        if len(groups) == 1:
            ver = next(iter(groups))
            render_fn(renumber_runs(groups[ver]), version_label(ver))
            return

    label = "Human" if view_mode == VIEW_HUMAN else "Agent"
    render_fn(renumber_runs(runs), label)


def render_dashboard(view_mode: str, runs: pd.DataFrame, render_fn) -> None:
    """Route charts/metrics to the correct layout for the current view + filters."""
    render_by_source(view_mode, runs, render_fn)


def win_rate_pct(df: pd.DataFrame) -> float | None:
    if df.empty or "won" not in df.columns:
        return None
    return float(df["won"].mean() * 100)


def combat_hp_pct_series(runs: pd.DataFrame) -> pd.Series:
    """Average HP% remaining after each combat, as 0-100 (run-level, length-normalized)."""
    if runs.empty or "avg_hp_pct_after_combat" not in runs.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(runs["avg_hp_pct_after_combat"], errors="coerce") * 100.0


def mean_combat_hp_pct(runs: pd.DataFrame) -> float | None:
    series = combat_hp_pct_series(runs).dropna()
    if series.empty:
        return None
    return float(series.mean())


def duration_from_decisions(decisions: pd.DataFrame) -> pd.Series:
    """Per-run wall-clock span from first to last logged decision (seconds)."""
    if decisions.empty or "run_id" not in decisions.columns or "timestamp" not in decisions.columns:
        return pd.Series(dtype=float)
    dec = decisions.dropna(subset=["timestamp"]).copy()
    if dec.empty:
        return pd.Series(dtype=float)
    spans = dec.groupby(dec["run_id"].astype(str))["timestamp"].agg(["min", "max"])
    return (spans["max"] - spans["min"]).dt.total_seconds()


def enrich_runs_with_duration(runs: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Attach run_duration_sec from runs.jsonl or infer from decisions.jsonl."""
    if runs.empty:
        return runs
    out = runs.copy()
    if "run_duration_sec" not in out.columns:
        out["run_duration_sec"] = pd.NA
    else:
        out["run_duration_sec"] = pd.to_numeric(out["run_duration_sec"], errors="coerce")

    inferred = duration_from_decisions(decisions)
    if inferred.empty:
        return out

    rid_str = out["run_id"].astype(str)
    for run_id, seconds in inferred.items():
        if pd.isna(seconds) or float(seconds) <= 0:
            continue
        mask = rid_str == str(run_id)
        missing = out.loc[mask, "run_duration_sec"].isna() | (out.loc[mask, "run_duration_sec"] <= 0)
        out.loc[mask & missing, "run_duration_sec"] = float(seconds)
    return out


def format_duration(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds) or float(seconds) <= 0:
        return "-"
    total = int(round(float(seconds)))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def mean_run_duration_sec(runs: pd.DataFrame) -> float | None:
    if runs.empty or "run_duration_sec" not in runs.columns:
        return None
    series = runs["run_duration_sec"].dropna()
    series = series[series > 0]
    if series.empty:
        return None
    return float(series.mean())


def resolve_date_filter(
    mode: str,
    runs: pd.DataFrame,
    *,
    custom_start: date | None = None,
    custom_end: date | None = None,
) -> tuple[bool, date | None, date | None]:
    """Map sidebar date preset to (active, inclusive start, inclusive end)."""
    if mode == "All time" or runs.empty or "timestamp" not in runs.columns:
        return False, None, None

    if mode == "Custom range":
        if custom_start is None or custom_end is None:
            return False, None, None
        return True, custom_start, custom_end

    days = DATE_PRESET_DAYS.get(mode)
    if days is None:
        return False, None, None

    end_ts = runs["timestamp"].max()
    start_d = (end_ts - pd.Timedelta(days=days)).date()
    return True, start_d, end_ts.date()


def compute_rolling_winrate(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Rolling win rate (0-100) from runs DataFrame."""
    if df.empty or "won" not in df.columns:
        return pd.Series(dtype=float)
    wins = df["won"].astype(int)
    return wins.rolling(window=window, min_periods=1).mean() * 100


def parse_death_category(cause: str | None) -> str:
    if not cause or not isinstance(cause, str):
        return "Unknown"
    lower = cause.lower()
    if "elite" in lower:
        return "Elite"
    if "boss" in lower:
        return "Boss"
    if "monster" in lower or "combat" in lower:
        return "Monster"
    if "event" in lower:
        return "Event"
    if "shop" in lower:
        return "Shop"
    if "interrupt" in lower:
        return "Interrupted"
    return "Other"


def extract_card_from_reasoning(text: str | None) -> str | None:
    """Card name from rules handler text (may appear after policy_net reasoning)."""
    if not text:
        return None
    match = re.search(r"best card:\s*([^(;]+)", text, re.I)
    if match:
        return match.group(1).strip()
    return None


def card_pick_label_from_row(row: pd.Series) -> str | None:
    """Label for a card_reward pick - rules name, else slot index (policy / BC)."""
    name = extract_card_from_reasoning(row.get("action_reasoning"))
    if name:
        return name
    text = str(row.get("action_reasoning") or "")
    match = re.search(r"key=select_card_reward:(\d+)", text)
    if match:
        return f"reward slot {match.group(1)}"
    idx = row.get("card_index")
    if pd.notna(idx):
        try:
            return f"reward slot {int(idx)}"
        except (TypeError, ValueError):
            pass
    return None


def card_played_name(row: pd.Series) -> str | None:
    action = row.get("action")
    if action != "play_card":
        return None
    hand = row.get("hand")
    idx = row.get("card_index")
    if not isinstance(hand, list) or pd.isna(idx):
        return None
    try:
        i = int(idx)
        if 0 <= i < len(hand) and isinstance(hand[i], dict):
            return str(hand[i].get("name") or hand[i].get("id") or f"card_{i}")
    except (TypeError, ValueError):
        pass
    return None


def current_agent_version(runs: pd.DataFrame) -> str | None:
    """Agent version with the most recent run (after non-version filters)."""
    agent = agent_runs_only(runs)
    if agent.empty or "agent_version" not in agent.columns:
        return None
    if "timestamp" in agent.columns and agent["timestamp"].notna().any():
        latest = agent.dropna(subset=["timestamp"]).sort_values("timestamp").iloc[-1]
        return str(latest["agent_version"])
    versions = available_agent_versions(runs)
    return versions[-1] if versions else None


def apply_filters(
    runs: pd.DataFrame,
    decisions: pd.DataFrame,
    card_choices: pd.DataFrame,
    sidebar: dict,
    *,
    agent_versions: list[str] | None = None,
    skip_version_filter: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r, d, cc = runs.copy(), decisions.copy(), card_choices.copy()
    if r.empty:
        return r, d, cc.iloc[0:0]

    view_mode = sidebar.get("view_mode", VIEW_COMPARE)
    if "source" in r.columns:
        if view_mode == VIEW_HUMAN:
            r = r[r["source"] == "human"]
        elif view_mode in (VIEW_AGENT, VIEW_COMPARE_VERSIONS):
            r = r[r["source"] != "human"]
        elif view_mode == VIEW_COMPARE:
            pass  # keep human + agent for side-by-side

    if skip_version_filter:
        selected_versions = []
    elif agent_versions is not None:
        selected_versions = list(agent_versions)
    else:
        selected_versions = sidebar.get("selected_versions") or []

    if selected_versions and "agent_version" in r.columns:
        if view_mode == VIEW_COMPARE and "source" in r.columns:
            r = r[(r["source"] == "human") | (r["agent_version"].isin(selected_versions))]
        elif view_mode in (VIEW_AGENT, VIEW_COMPARE_VERSIONS):
            r = r[r["agent_version"].isin(selected_versions)]
    if sidebar.get("character") and sidebar["character"] != "All":
        r = r[r["character"] == sidebar["character"]]
    if sidebar.get("ascension") is not None and sidebar["ascension"] != "All":
        r = r[r["ascension"] == int(sidebar["ascension"])]
    if sidebar.get("filter_dates") and "timestamp" in r.columns:
        if sidebar.get("date_start"):
            r = r[r["timestamp"] >= pd.Timestamp(sidebar["date_start"], tz="UTC")]
        if sidebar.get("date_end"):
            end = pd.Timestamp(sidebar["date_end"], tz="UTC") + timedelta(days=1)
            r = r[r["timestamp"] < end]

    run_ids = set(r["run_id"].astype(str)) if not r.empty and "run_id" in r.columns else set()

    if not d.empty and "run_id" in d.columns:
        d = d[d["run_id"].astype(str).isin(run_ids)] if run_ids else d.iloc[0:0]
    elif not r.empty:
        d = d.iloc[0:0]

    if not cc.empty and "run_id" in cc.columns:
        cc = cc[cc["run_id"].astype(str).isin(run_ids)] if run_ids else cc.iloc[0:0]
    elif not r.empty:
        cc = cc.iloc[0:0]

    return r.reset_index(drop=True), d.reset_index(drop=True), cc.reset_index(drop=True)


def empty_chart(title: str, message: str = "No data yet") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": "#888"},
    )
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


def render_sidebar(runs: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")

    if st.sidebar.button("Reload data"):
        st.cache_data.clear()
        st.rerun()

    _ensure_select_value(FILTER_VIEW_MODE, list(VIEW_MODES), VIEW_COMPARE)
    view_mode = st.sidebar.selectbox(
        "View",
        VIEW_MODES,
        key=FILTER_VIEW_MODE,
        help=(
            "Human vs agent, or compare agent versions side by side. "
            "Use the version checklist in the sidebar to choose which tags to include."
        ),
    )

    selected_versions: list[str] = []
    if view_mode in (VIEW_AGENT, VIEW_COMPARE, VIEW_COMPARE_VERSIONS):
        versions = available_agent_versions(runs)
        if versions:
            st.sidebar.markdown("**Agent versions**")
            version_defaults = _saved_agent_versions(versions)

            c_all, c_none = st.sidebar.columns(2)
            if c_all.button("Select all", use_container_width=True):
                st.session_state[FILTER_SAVED_AGENT_VERSIONS] = _default_agent_versions(versions)
                st.rerun()
            if c_none.button("Clear", use_container_width=True):
                st.session_state[FILTER_SAVED_AGENT_VERSIONS] = []
                st.rerun()

            selected_versions = st.sidebar.multiselect(
                "Versions for overview (section 1)",
                options=versions,
                default=version_defaults,
                help=(
                    "Multi-version health trends and early warnings (section 1 only). "
                    "Sections 2–4 use the analyzer picker below the overview. "
                    "BC tags (bc_v*) are unchecked by default."
                ),
            )
            st.session_state[FILTER_SAVED_AGENT_VERSIONS] = list(selected_versions)
            if not selected_versions:
                st.sidebar.warning("No versions selected - agent panels will be empty.")
        elif view_mode != VIEW_COMPARE:
            st.sidebar.caption("No agent versions in dataset yet.")
    elif view_mode == VIEW_HUMAN:
        st.sidebar.caption("Human imports only - no agent versions.")

    characters = ["All"]
    ascensions = ["All"]
    if not runs.empty:
        if "character" in runs.columns:
            characters += sorted(runs["character"].dropna().unique().tolist())
        if "ascension" in runs.columns:
            ascensions += [str(a) for a in sorted(runs["ascension"].dropna().unique())]

    char_default = (
        DEFAULT_DASHBOARD_CHARACTER
        if DEFAULT_DASHBOARD_CHARACTER in characters
        else "All"
    )
    _ensure_select_value(FILTER_CHARACTER, characters, char_default)
    _ensure_select_value(FILTER_ASCENSION, ascensions, "All")
    character = st.sidebar.selectbox("Character", characters, key=FILTER_CHARACTER)
    ascension = st.sidebar.selectbox("Ascension", ascensions, key=FILTER_ASCENSION)

    filter_dates = False
    date_start, date_end = None, None
    date_mode = "All time"
    if not runs.empty and "timestamp" in runs.columns:
        min_d = runs["timestamp"].min().date()
        max_d = runs["timestamp"].max().date()
        date_mode = st.sidebar.selectbox("Date filter", DATE_FILTER_OPTIONS, index=0)

        if date_mode == "All time":
            st.sidebar.caption(f"All runs · {min_d} → {max_d}")
        elif date_mode == "Custom range":
            dr = st.sidebar.date_input(
                "Date range",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )
            custom_start, custom_end = None, None
            if isinstance(dr, tuple) and len(dr) == 2:
                custom_start, custom_end = dr[0], dr[1]
            elif hasattr(dr, "year"):
                custom_start = custom_end = dr
            filter_dates, date_start, date_end = resolve_date_filter(
                date_mode,
                runs,
                custom_start=custom_start,
                custom_end=custom_end,
            )
        else:
            filter_dates, date_start, date_end = resolve_date_filter(date_mode, runs)

        if filter_dates and date_start and date_end:
            st.sidebar.caption(f"Showing {date_start} → {date_end}")
        if date_mode == "All time" and "source" in runs.columns:
            human = runs[runs["source"] == "human"]
            agent = runs[runs["source"] != "human"]
            if not human.empty:
                h0, h1 = human["timestamp"].min().date(), human["timestamp"].max().date()
                st.sidebar.caption(f"Human: {h0} → {h1}")
            if not agent.empty:
                a0, a1 = agent["timestamp"].min().date(), agent["timestamp"].max().date()
                st.sidebar.caption(f"Agent: {a0} → {a1}")

    st.sidebar.divider()
    if st.sidebar.button("Clear all data", type="primary"):
        st.session_state["confirm_clear"] = True

    if st.session_state.get("confirm_clear"):
        st.sidebar.warning("Delete all training data?")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Yes, delete"):
            for path in (RUNS_PATH, DECISIONS_PATH):
                if path.exists():
                    path.unlink()
            st.session_state["confirm_clear"] = False
            st.cache_data.clear()
            st.sidebar.success("Data cleared.")
            st.rerun()
        if c2.button("Cancel"):
            st.session_state["confirm_clear"] = False
            st.rerun()

    if not runs.empty:
        n_human = int((runs["source"] == "human").sum()) if "source" in runs.columns else 0
        n_agent = int((runs["source"] != "human").sum()) if "source" in runs.columns else len(runs)
        caption = f"Loaded: {len(runs)} runs ({n_human} human, {n_agent} agent)"
        ver_caption = version_count_caption(agent_runs_only(runs))
        if ver_caption:
            caption += f" - {ver_caption}"
        if selected_versions and view_mode in (VIEW_AGENT, VIEW_COMPARE, VIEW_COMPARE_VERSIONS):
            caption += f" · section 1: {len(selected_versions)} version(s)"
        st.sidebar.caption(caption)

    return {
        "view_mode": view_mode,
        "character": character,
        "ascension": ascension,
        "selected_versions": selected_versions,
        "date_mode": date_mode,
        "filter_dates": filter_dates,
        "date_start": date_start,
        "date_end": date_end,
    }


def main() -> None:
    st.set_page_config(
        page_title="STS2 Agent Dashboard",
        page_icon="🃏",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    from dashboard.sections import (
        section_cards,
        section_combat_efficiency,
        section_death,
        section_health,
    )

    revision = _data_revision_key()
    runs_raw = load_runs(revision)
    decisions_raw = load_decisions(revision)
    card_choices_raw = load_card_choices(revision)
    runs_raw = enrich_runs_with_duration(runs_raw, decisions_raw)
    sidebar = render_sidebar(runs_raw)

    runs_overview, _, _ = apply_filters(
        runs_raw, decisions_raw, card_choices_raw, sidebar
    )

    runs_scoped, _, _ = apply_filters(
        runs_raw,
        decisions_raw,
        card_choices_raw,
        sidebar,
        skip_version_filter=True,
    )
    detail_versions = available_agent_versions(runs_scoped)
    default_detail = current_agent_version(runs_scoped)
    if default_detail and default_detail not in detail_versions:
        detail_versions = sorted(set(detail_versions) | {default_detail})

    tab_runs, tab_compendium = st.tabs(["Run analytics", "Enemy compendium"])

    with tab_runs:
        st.title("STS2 Agent Dashboard")
        st.caption("Phase A analytics — reload sidebar after new runs")
        section_health(runs_overview)
        st.divider()

        from dashboard.sections import render_detail_version_picker

        detail_version = render_detail_version_picker(detail_versions, default_detail)
        if detail_version:
            runs_detail, decisions_detail, card_choices_detail = apply_filters(
                runs_raw,
                decisions_raw,
                card_choices_raw,
                sidebar,
                agent_versions=[detail_version],
            )
        else:
            runs_detail, decisions_detail, card_choices_detail = (
                runs_scoped.iloc[0:0],
                decisions_raw.iloc[0:0],
                card_choices_raw.iloc[0:0],
            )
            st.info("No agent versions in dataset for detailed analysis.")
            phase_b_count, phase_b_total = 0, 0

        if detail_version:
            from dashboard.metrics import filter_detail_phase_b

            (
                runs_detail,
                decisions_detail,
                card_choices_detail,
                phase_b_count,
                phase_b_total,
            ) = filter_detail_phase_b(
                runs_detail, decisions_detail, card_choices_detail
            )

        section_death(
            runs_detail,
            decisions_detail,
            agent_version=detail_version,
            phase_b_count=phase_b_count,
            phase_b_total=phase_b_total,
        )
        st.divider()
        section_cards(
            runs_detail,
            decisions_detail,
            card_choices_detail,
            agent_version=detail_version,
            phase_b_count=phase_b_count,
            phase_b_total=phase_b_total,
        )
        st.divider()
        section_combat_efficiency(
            runs_detail,
            decisions_detail,
            agent_version=detail_version,
            phase_b_count=phase_b_count,
            phase_b_total=phase_b_total,
        )

        st.caption(
            f"Data: `{RUNS_PATH.relative_to(PROJECT_ROOT)}` · "
            f"`{DECISIONS_PATH.relative_to(PROJECT_ROOT)}` · "
            f"`{CARD_CHOICES_PATH.relative_to(PROJECT_ROOT)}` · "
            f"Last load: {datetime.now().strftime('%H:%M:%S')}"
        )

    with tab_compendium:
        if str(DASHBOARD_DIR) not in sys.path:
            sys.path.insert(0, str(DASHBOARD_DIR))
        from enemy_compendium import render_compendium

        render_compendium()


if __name__ == "__main__":
    main()
