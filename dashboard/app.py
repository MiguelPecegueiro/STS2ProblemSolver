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

# Streamlit session_state keys - persist sidebar filters across "Reload data".
FILTER_VIEW_MODE = "dash_filter_view_mode"
FILTER_CHARACTER = "dash_filter_character"
FILTER_ASCENSION = "dash_filter_ascension"
FILTER_DATE_MODE = "dash_filter_date_mode"
FILTER_CUSTOM_DATES = "dash_filter_custom_dates"
FILTER_SAVED_AGENT_VERSIONS = "dash_saved_agent_versions"
FILTER_DECISION_RUN = "dash_filter_decision_run"


def _ensure_select_value(key: str, options: list[str], default: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = default
        return
    if st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]


def _saved_agent_versions(versions: list[str]) -> list[str]:
    legacy_key = "agent_versions_pick"
    if FILTER_SAVED_AGENT_VERSIONS not in st.session_state:
        if legacy_key in st.session_state:
            st.session_state[FILTER_SAVED_AGENT_VERSIONS] = list(st.session_state[legacy_key])
        else:
            st.session_state[FILTER_SAVED_AGENT_VERSIONS] = list(versions)
    saved = [v for v in st.session_state[FILTER_SAVED_AGENT_VERSIONS] if v in versions]
    if saved:
        return saved
    if not versions:
        return []
    # First visit only: default to all versions (do not reset after user clears selection).
    if st.session_state[FILTER_SAVED_AGENT_VERSIONS] == []:
        return []
    return list(versions)


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
        action = row.get("action_taken") or {}
        outcome = row.get("run_outcome") or {}
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
                "immediate_reward": row.get("immediate_reward"),
                "run_won": outcome.get("won") if isinstance(outcome, dict) else None,
                "agent_version": row.get("agent_version"),
                "player_hp": snap.get("player_hp"),
                "player_max_hp": snap.get("player_max_hp"),
                "hand": snap.get("hand"),
            }
        )
    df = pd.DataFrame(flat)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    for col in ("floor", "act", "immediate_reward", "card_index"):
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


def apply_filters(
    runs: pd.DataFrame,
    decisions: pd.DataFrame,
    card_choices: pd.DataFrame,
    sidebar: dict,
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
                st.session_state[FILTER_SAVED_AGENT_VERSIONS] = list(versions)
                st.rerun()
            if c_none.button("Clear", use_container_width=True):
                st.session_state[FILTER_SAVED_AGENT_VERSIONS] = []
                st.rerun()

            selected_versions = st.sidebar.multiselect(
                "Versions to show",
                options=versions,
                default=version_defaults,
                help="Charts and tables only include checked agent versions",
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

    _ensure_select_value(FILTER_CHARACTER, characters, "All")
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
            caption += f" · showing {len(selected_versions)} version(s)"
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


def _version_summary_table(runs: pd.DataFrame) -> None:
    groups = agent_version_groups(runs)
    if len(groups) < 2:
        return

    rows: list[dict] = []
    for ver, df in sorted(groups.items()):
        wr = win_rate_pct(df) or 0.0
        row: dict = {
            "Version": version_label(ver),
            "Runs": len(df),
            "Win rate": f"{wr:.1f}%",
            "Avg floor": round(float(df["floors_reached"].mean()), 1),
            "Avg act": round(float(df["act_reached"].mean()), 1),
            "Wins": int(df["won"].sum()),
        }
        hp_mean = mean_combat_hp_pct(df)
        if hp_mean is not None:
            row["Avg HP% combat"] = f"{hp_mean:.1f}%"
        dur_mean = mean_run_duration_sec(df)
        if dur_mean is not None:
            row["Avg duration"] = format_duration(dur_mean)
        if "run_score" in df.columns and df["run_score"].sum() > 0:
            row["Avg score"] = round(float(df["run_score"].mean()), 0)
        rows.append(row)

    st.markdown("**Agent version comparison**")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _metrics_block(runs: pd.DataFrame, label: str | None) -> None:
    if label:
        st.markdown(f"##### {label}")
    if runs.empty:
        st.caption("No runs for these filters.")
        return

    wr = win_rate_pct(runs) or 0.0
    last10 = runs.tail(10)
    prev10 = runs.iloc[-20:-10] if len(runs) >= 20 else runs.iloc[: max(0, len(runs) - 10)]
    wr_last = last10["won"].mean() * 100 if len(last10) else 0
    wr_prev = prev10["won"].mean() * 100 if len(prev10) else wr_last
    delta_wr = wr_last - wr_prev

    c1, c2, c3 = st.columns(3)
    c1.metric("Runs", len(runs))
    c2.metric("Win Rate", f"{wr:.1f}%", delta=f"{delta_wr:+.1f}% vs prior 10")
    c3.metric("Avg Floor", f"{runs['floors_reached'].mean():.1f}")

    dur_mean = mean_run_duration_sec(runs)
    has_score = "run_score" in runs.columns and runs["run_score"].sum() > 0
    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Avg Act", f"{runs['act_reached'].mean():.1f}")
    if dur_mean is not None:
        c5.metric(
            "Avg run duration",
            format_duration(dur_mean),
            help="Wall-clock time from first to last logged decision, or pipeline timer for new runs.",
        )
    else:
        c5.caption("No duration data")
    if has_score:
        c6.metric("Avg run score", f"{runs['run_score'].mean():.0f}")
    else:
        c6.caption("-")
    c7.metric("Wins", int(runs["won"].sum()))


def section_metrics(
    runs: pd.DataFrame,
    runs_all: pd.DataFrame,
    view_mode: str,
    sidebar: dict,
) -> None:
    st.header("STS2 Agent Dashboard")
    st.caption("Training logs only - hit **Reload data** in the sidebar after new runs")

    if runs_all is not None and not runs_all.empty and "source" in runs_all.columns:
        n_human = int((runs_all["source"] == "human").sum())
        n_agent = int((runs_all["source"] != "human").sum())
        ver_caption = version_count_caption(agent_runs_only(runs_all))
        if view_mode == VIEW_COMPARE and n_human and n_agent:
            st.info(
                f"Human ({n_human}) vs agent ({n_agent}) runs. {ver_caption or ''}".strip()
            )
        elif view_mode == VIEW_HUMAN:
            st.info("Imported `.run` files - reward picks from card_choices.jsonl.")
        elif view_mode == VIEW_COMPARE_VERSIONS:
            groups = agent_version_groups(runs)
            if groups:
                st.info(
                    "Versions: " + ", ".join(version_label(v) for v in groups) + "."
                )
        elif view_mode == VIEW_AGENT:
            groups = agent_version_groups(runs)
            if len(groups) > 1:
                st.info(version_count_caption(runs) + ".")

    agent_for_summary = agent_runs_only(runs)
    if view_mode in (VIEW_AGENT, VIEW_COMPARE_VERSIONS) or (
        view_mode == VIEW_COMPARE and len(agent_version_groups(agent_for_summary)) > 1
    ):
        _version_summary_table(agent_for_summary)

    if view_mode in (VIEW_AGENT, VIEW_COMPARE_VERSIONS) and not agent_for_summary.empty:
        sel = sidebar.get("selected_versions") or []
        if not sel:
            st.warning("Select at least one agent version in the sidebar to see agent metrics.")

    render_dashboard(view_mode, runs, _metrics_block)


def _chart_win_rate(runs: pd.DataFrame, label: str | None) -> None:
    if label:
        st.markdown(f"##### {label}")
    if runs.empty:
        st.plotly_chart(empty_chart("Rolling win rate"), use_container_width=True)
        return

    rolling = compute_rolling_winrate(runs, 10)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=runs["run_number"],
            y=runs["won"].astype(int) * 100,
            mode="lines+markers",
            name="Per-run",
            line={"color": "rgba(150,150,150,0.5)", "width": 1},
            marker={"size": 4, "color": "rgba(150,150,150,0.6)"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=runs["run_number"],
            y=rolling,
            mode="lines",
            name="10-run rolling avg",
            line={"color": "#e74c3c", "width": 3},
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        xaxis_title="Run #",
        yaxis_title="Win rate %",
        yaxis={"range": [0, 105]},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_win_rate_overlay(runs: pd.DataFrame) -> None:
    """Single chart with one rolling win-rate line per agent_version."""
    groups = agent_version_groups(runs)
    if len(groups) < 2:
        return

    palette = px.colors.qualitative.Set1
    fig = go.Figure()
    for i, (ver, subset) in enumerate(sorted(groups.items())):
        ordered = renumber_runs(subset)
        rolling = compute_rolling_winrate(ordered, 10)
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=ordered["run_number"],
                y=rolling,
                mode="lines",
                name=version_label(ver),
                line={"color": color, "width": 2.5},
            )
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        title="Rolling win rate by agent version (10-run window)",
        xaxis_title="Run # (per version)",
        yaxis_title="Win rate %",
        yaxis={"range": [0, 105]},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def section_win_rate(runs: pd.DataFrame, view_mode: str, sidebar: dict) -> None:
    st.subheader("Win Rate Over Time")
    if view_mode in (VIEW_COMPARE_VERSIONS, VIEW_AGENT) and len(agent_version_groups(runs)) > 1:
        _chart_win_rate_overlay(runs)
    render_dashboard(view_mode, runs, _chart_win_rate)


def _chart_death(runs: pd.DataFrame, label: str | None) -> None:
    if label:
        st.markdown(f"##### {label}")
    left, right = st.columns(2)
    with left:
        if runs.empty:
            st.plotly_chart(empty_chart("Floors reached"), use_container_width=True)
        else:
            floor_counts = runs["floors_reached"].value_counts().sort_index()
            fig = px.bar(
                x=floor_counts.index.astype(str),
                y=floor_counts.values,
                labels={"x": "Floor", "y": "Runs ended"},
                title="Floors reached",
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(height=CHART_HEIGHT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if runs.empty:
            st.plotly_chart(empty_chart("Cause of death"), use_container_width=True)
        else:
            cats = runs["cause_of_death"].apply(parse_death_category)
            pie_df = cats.value_counts().reset_index()
            pie_df.columns = ["category", "count"]
            fig = px.pie(
                pie_df,
                names="category",
                values="count",
                title="Cause of death",
                template=PLOTLY_TEMPLATE,
                hole=0.35,
            )
            fig.update_layout(height=CHART_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)


def section_death(runs: pd.DataFrame, view_mode: str, sidebar: dict) -> None:
    st.subheader("Where Does It Die?")
    render_dashboard(view_mode, runs, _chart_death)


def _chart_act_progression(runs: pd.DataFrame, label: str | None) -> None:
    if label:
        st.markdown(f"##### {label}")
    if runs.empty:
        st.plotly_chart(empty_chart("Act progression"), use_container_width=True)
        return

    def bucket(row: pd.Series) -> str:
        if row.get("won"):
            return "Won"
        act = int(row.get("act_reached") or 1)
        return f"Act {act} only"

    subset = runs.copy()
    subset["bucket"] = subset.apply(bucket, axis=1)
    order = ["Act 1 only", "Act 2 only", "Act 3 only", "Won"]
    counts = subset["bucket"].value_counts()
    counts = counts.reindex([o for o in order if o in counts.index]).fillna(0)

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color=["#3498db", "#9b59b6", "#e67e22", "#2ecc71"][: len(counts)],
            )
        ]
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        title="Highest act / victory",
        xaxis_title="Outcome",
        yaxis_title="Runs",
    )
    st.plotly_chart(fig, use_container_width=True)


def section_act_progression(runs: pd.DataFrame, view_mode: str, sidebar: dict) -> None:
    st.subheader("Act Progression")
    render_dashboard(view_mode, runs, _chart_act_progression)


def _chart_combat(runs: pd.DataFrame, label: str | None) -> None:
    if label:
        st.markdown(f"##### {label}")
    hp_pct = combat_hp_pct_series(runs)
    mean_hp = mean_combat_hp_pct(runs)

    left, right = st.columns(2)
    with left:
        if mean_hp is not None:
            st.metric(
                "Avg HP% after combat",
                f"{mean_hp:.1f}%",
                help="Mean HP remaining after each fight, averaged over the run (0-100%). "
                "Comparable across runs that end at different floors.",
            )
        if runs.empty or hp_pct.dropna().empty:
            st.plotly_chart(
                empty_chart(
                    "HP% after combat per run",
                    "No avg_hp_pct_after_combat in runs - re-import or run the agent with current pipeline",
                ),
                use_container_width=True,
            )
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=runs["run_number"],
                    y=hp_pct,
                    mode="lines+markers",
                    name="HP% after combat",
                    line={"color": "#2ecc71"},
                    connectgaps=False,
                )
            )
            if mean_hp is not None:
                fig.add_hline(
                    y=mean_hp,
                    line_dash="dash",
                    line_color="#95a5a6",
                    annotation_text=f"mean {mean_hp:.1f}%",
                    annotation_position="right",
                )
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=CHART_HEIGHT,
                title="HP% after combat per run",
                xaxis_title="Run #",
                yaxis_title="Avg HP% after combat",
                yaxis={"range": [0, 100]},
            )
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if runs.empty:
            st.plotly_chart(empty_chart("Damage dealt"), use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=runs["run_number"],
                    y=runs["total_damage_dealt"],
                    mode="lines+markers",
                    name="Damage dealt",
                    line={"color": "#3498db"},
                )
            )
            if runs["total_damage_dealt"].sum() == 0:
                fig.add_annotation(
                    text="No damage-dealt tracking in pipeline",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font={"size": 12, "color": "#888"},
                )
            fig.update_layout(
                template=PLOTLY_TEMPLATE,
                height=CHART_HEIGHT,
                title="Damage dealt per run",
                xaxis_title="Run #",
                yaxis_title="Total damage dealt",
            )
            st.plotly_chart(fig, use_container_width=True)


def _chart_run_timing(runs: pd.DataFrame, label: str | None) -> None:
    if label:
        st.markdown(f"##### {label}")
    dur_mean = mean_run_duration_sec(runs)
    if dur_mean is not None:
        st.metric(
            "Avg run duration",
            format_duration(dur_mean),
            help="Comparable across versions - longer runs are not penalized in this average.",
        )

    if runs.empty or "run_duration_sec" not in runs.columns:
        st.plotly_chart(empty_chart("Run duration"), use_container_width=True)
        return

    durations = runs["run_duration_sec"].dropna()
    durations = durations[durations > 0]
    if durations.empty:
        st.plotly_chart(
            empty_chart(
                "Run duration",
                "No timing data - agent runs need decisions.jsonl or run_duration_sec in runs.jsonl",
            ),
            use_container_width=True,
        )
        return

    plot_df = runs.loc[durations.index].copy()
    plot_df["duration_min"] = plot_df["run_duration_sec"] / 60.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["run_number"],
            y=plot_df["duration_min"],
            mode="lines+markers",
            name="Duration",
            line={"color": "#9b59b6"},
            connectgaps=False,
        )
    )
    if dur_mean is not None:
        fig.add_hline(
            y=dur_mean / 60.0,
            line_dash="dash",
            line_color="#95a5a6",
            annotation_text=f"mean {format_duration(dur_mean)}",
            annotation_position="right",
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        title="Run duration",
        xaxis_title="Run #",
        yaxis_title="Minutes",
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_run_timing_by_version(runs: pd.DataFrame) -> None:
    """Bar chart of mean duration per agent_version (filtered agent runs)."""
    agent = agent_runs_only(runs)
    if agent.empty or "agent_version" not in agent.columns:
        st.info("No version-tagged agent runs.")
        return
    groups = agent_version_groups(agent)
    rows: list[dict] = []
    for ver, df in groups.items():
        mean_sec = mean_run_duration_sec(df)
        if mean_sec is None:
            continue
        rows.append(
            {
                "version": version_label(ver),
                "duration_min": mean_sec / 60.0,
                "runs": len(df),
            }
        )
    if not rows:
        st.info("No duration data yet.")
        return
    bar_df = pd.DataFrame(rows)
    fig = px.bar(
        bar_df,
        x="version",
        y="duration_min",
        title="Average run duration by agent version",
        labels={"duration_min": "Avg minutes", "version": "Agent version"},
        template=PLOTLY_TEMPLATE,
        text=bar_df["runs"].apply(lambda n: f"{n} runs"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=CHART_HEIGHT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def section_run_timing(runs: pd.DataFrame, view_mode: str, sidebar: dict) -> None:
    st.subheader("Run timing")
    st.caption(
        "Run length from runs.jsonl timestamps, or first/last decision row if missing."
    )
    if view_mode == VIEW_COMPARE_VERSIONS:
        _chart_run_timing_by_version(runs)
        st.divider()
    render_dashboard(view_mode, runs, _chart_run_timing)


def section_combat(runs: pd.DataFrame, decisions: pd.DataFrame, view_mode: str, sidebar: dict) -> None:
    st.subheader("Combat")
    st.caption("Avg HP% left after each fight (per run). Comparable across short and long runs.")
    render_dashboard(view_mode, runs, _chart_combat)
    if view_mode == VIEW_HUMAN:
        st.caption("Combat plays need agent decisions.jsonl.")
    elif view_mode == VIEW_COMPARE:
        st.caption("HP% per source; damage on the right. Card picks split by agent version.")


def _agent_reward_picks(decisions: pd.DataFrame) -> list[str]:
    if decisions.empty:
        return []
    picks = decisions[
        (decisions["state_type"] == "card_reward")
        & (decisions["action"] == "select_card_reward")
    ]
    return picks.apply(card_pick_label_from_row, axis=1).dropna().astype(str).tolist()


def _chart_card_picks(pick_names: list[str], title: str) -> None:
    top = Counter(pick_names).most_common(10)
    if not top:
        st.plotly_chart(empty_chart("Card picks", "No reward picks"), use_container_width=True)
        return
    pdf = pd.DataFrame(top, columns=["card", "picks"])
    fig = px.bar(
        pdf,
        x="picks",
        y="card",
        orientation="h",
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=CHART_HEIGHT, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def _chart_combat_plays(decisions: pd.DataFrame, title: str) -> None:
    if decisions.empty:
        st.plotly_chart(empty_chart("Combat plays", "Agent runs only"), use_container_width=True)
        return
    combat = decisions[decisions["state_type"].isin(["monster", "elite", "boss", "hand_select"])]
    played = combat.apply(card_played_name, axis=1).dropna()
    top = Counter(played).most_common(10)
    if not top:
        st.plotly_chart(empty_chart("Combat plays", "No plays logged yet"), use_container_width=True)
        return
    pdf = pd.DataFrame(top, columns=["card", "plays"])
    fig = px.bar(
        pdf,
        x="plays",
        y="card",
        orientation="h",
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(height=CHART_HEIGHT, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def _card_intel_agent_panel(decisions: pd.DataFrame, label: str) -> None:
    """Reward picks + combat plays for one agent_version."""
    st.markdown(f"##### {label}")
    if decisions.empty:
        st.caption("No decisions for this version.")
        return
    left, right = st.columns(2)
    with left:
        pick_names = _agent_reward_picks(decisions)
        _chart_card_picks(
            pick_names,
            f"Top 10 reward picks ({len(pick_names)})",
        )
    with right:
        _chart_combat_plays(decisions, "Top 10 combat plays")


def render_card_intel_by_agent(decisions: pd.DataFrame, runs: pd.DataFrame) -> None:
    groups = agent_decisions_by_version(decisions, runs)
    if not groups:
        st.caption("No agent decisions for these filters.")
        return
    if len(groups) == 1:
        ver, subset = next(iter(groups.items()))
        _card_intel_agent_panel(subset, version_label(ver))
        return
    versions = list(groups.keys())
    cols = st.columns(len(versions))
    for col, ver in zip(cols, versions):
        with col:
            _card_intel_agent_panel(groups[ver], version_label(ver))


def section_cards(
    decisions: pd.DataFrame,
    card_choices: pd.DataFrame,
    runs: pd.DataFrame,
    view_mode: str,
    *,
    filter_label: str | None = None,
) -> None:
    st.subheader("Card picks & plays")
    if filter_label:
        st.caption(filter_label)
    if view_mode in (VIEW_AGENT, VIEW_COMPARE_VERSIONS):
        st.caption("One column per agent version.")
    elif view_mode == VIEW_COMPARE:
        st.caption("Human on the left; agent versions on the right.")

    if view_mode == VIEW_COMPARE:
        col_h, col_a = st.columns(2)
        with col_h:
            st.markdown("##### Human - reward picks")
            human_names = (
                card_choices["picked"].dropna().astype(str).tolist()
                if not card_choices.empty and "picked" in card_choices.columns
                else []
            )
            _chart_card_picks(human_names, f"Top 10 ({len(human_names)} picks)")
        with col_a:
            render_card_intel_by_agent(decisions, runs)
        return

    if view_mode == VIEW_HUMAN:
        left, right = st.columns(2)
        with left:
            human_names = (
                card_choices["picked"].dropna().astype(str).tolist()
                if not card_choices.empty and "picked" in card_choices.columns
                else []
            )
            _chart_card_picks(human_names, f"Top 10 human reward picks ({len(human_names)})")
        with right:
            st.plotly_chart(
                empty_chart("Combat plays", "Not available for human imports"),
                use_container_width=True,
            )
        return

    render_card_intel_by_agent(decisions, runs)


def _deck_stats_block(runs: pd.DataFrame, label: str | None) -> None:
    if label:
        st.markdown(f"##### {label}")
    if runs.empty:
        st.caption("No runs for these filters.")
        return

    st.metric("Avg deck size at death", f"{runs['deck_size'].mean():.1f}")
    wins = runs[runs["won"]]
    losses = runs[~runs["won"]]
    left, right = st.columns(2)

    def deck_counter(subset: pd.DataFrame) -> Counter:
        c: Counter = Counter()
        for deck in subset.get("final_deck", []):
            if isinstance(deck, list):
                c.update(deck)
        return c

    with left:
        st.markdown("**Winning runs - top cards**")
        wc = deck_counter(wins).most_common(8)
        if wc:
            st.table(pd.DataFrame(wc, columns=["card", "count"]))
        else:
            st.caption("No winning runs yet.")
    with right:
        st.markdown("**Losing runs - top cards**")
        lc = deck_counter(losses).most_common(8)
        if lc:
            st.table(pd.DataFrame(lc, columns=["card", "count"]))
        else:
            st.caption("No data.")
    relics: Counter = Counter()
    for rel_list in wins.get("final_relics", []):
        if isinstance(rel_list, list):
            relics.update(rel_list)
    if relics:
        st.caption("Relics in wins: " + ", ".join(f"{n} ({c})" for n, c in relics.most_common(8)))


def section_deck_stats(runs: pd.DataFrame, view_mode: str, sidebar: dict) -> None:
    st.subheader("Deck Stats")
    if view_mode in (VIEW_COMPARE, VIEW_COMPARE_VERSIONS, VIEW_AGENT):
        render_dashboard(view_mode, runs, _deck_stats_block)
        return
    if runs.empty:
        st.info("No runs for these filters.")
        return
    _deck_stats_block(runs, None)


def _recent_runs_table(runs: pd.DataFrame, label: str | None) -> None:
    if label:
        st.markdown(f"##### {label}")
    if runs.empty:
        st.caption("No runs.")
        return

    display = runs.tail(20).iloc[::-1].copy()
    display["run_id_short"] = display["run_id"].astype(str).str[:8] + "…"
    display["result"] = display["won"].map({True: "Win", False: "Loss"})
    display["when"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

    cols = [
        "run_id_short",
        "when",
        "result",
        "floors_reached",
        "act_reached",
        "cause_of_death",
        "deck_size",
        "total_damage_taken",
    ]
    if "run_duration_sec" in display.columns and display["run_duration_sec"].notna().any():
        cols.insert(4, "run_duration_sec")
        display["run_duration_sec"] = display["run_duration_sec"].apply(
            lambda s: format_duration(float(s)) if pd.notna(s) and float(s) > 0 else "-"
        )
    rename = {
        "run_id_short": "Run ID",
        "when": "Time",
        "result": "Result",
        "floors_reached": "Floor",
        "act_reached": "Act",
        "cause_of_death": "Cause of death",
        "deck_size": "Deck size",
        "total_damage_taken": "Dmg taken",
        "run_duration_sec": "Duration",
    }
    if "run_score" in display.columns:
        cols.insert(3, "run_score")
        rename["run_score"] = "Score"
    if "agent_version" in display.columns:
        cols.insert(3, "agent_version")
        rename["agent_version"] = "Version"

    table = display[cols].rename(columns=rename)

    def row_style(row: pd.Series) -> list[str]:
        if row.get("Result") == "Win":
            return ["background-color: rgba(46, 204, 113, 0.15)"] * len(row)
        return ["background-color: rgba(231, 76, 60, 0.12)"] * len(row)

    styled = table.style.apply(row_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def section_recent_runs(runs: pd.DataFrame, view_mode: str, sidebar: dict) -> None:
    st.subheader("Recent Runs")
    if view_mode in (VIEW_COMPARE, VIEW_COMPARE_VERSIONS, VIEW_AGENT):
        render_dashboard(view_mode, runs, _recent_runs_table)
        return
    if runs.empty:
        st.info("No runs for these filters.")
        return
    _recent_runs_table(runs, None)


def section_decision_explorer(
    runs: pd.DataFrame,
    decisions: pd.DataFrame,
    card_choices: pd.DataFrame,
) -> None:
    st.subheader("Decision Explorer")
    if runs.empty:
        st.info("No runs in dataset.")
        return

    def run_label(row: pd.Series) -> str:
        src = row.get("source", "agent")
        if src == "human":
            tag = "human"
        else:
            ver = row.get("agent_version", "")
            tag = f"agent/{ver}" if ver else "agent"
        rid = str(row["run_id"])
        short = rid[:8] + "…" if len(rid) > 8 else rid
        result = "Win" if row["won"] else "Loss"
        return f"[{tag}] {short} - {result} - floor {row['floors_reached']}"

    options = {run_label(row): row["run_id"] for _, row in runs.iloc[::-1].iterrows()}
    run_labels = list(options.keys())
    if not run_labels:
        return
    if FILTER_DECISION_RUN not in st.session_state or st.session_state[FILTER_DECISION_RUN] not in run_labels:
        st.session_state[FILTER_DECISION_RUN] = run_labels[0]
    label = st.selectbox("Select run", run_labels, key=FILTER_DECISION_RUN)
    run_id = options[label]

    run_row = runs[runs["run_id"] == run_id].iloc[0]
    run_dec = decisions[decisions["run_id"] == run_id].sort_values(["floor", "timestamp"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Result", "Win" if run_row["won"] else "Loss")
    c2.metric("Floors", int(run_row["floors_reached"]))
    c3.metric("Decisions", len(run_dec))
    c4.metric("Damage taken", int(run_row["total_damage_taken"]))

    if run_dec.empty:
        human_picks = card_choices[card_choices["run_id"] == run_id] if not card_choices.empty else pd.DataFrame()
        if not human_picks.empty:
            st.markdown("**Human card reward picks** (from imported .run file)")
            st.dataframe(
                human_picks[
                    ["floor", "act", "floor_type", "offered", "picked", "hp_at_pick"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning(
                "No agent decisions logged for this run. "
                "Human imports do not populate decisions.jsonl."
            )
        return

    timeline = run_dec[
        ["floor", "act", "state_type", "action", "immediate_reward", "action_reasoning"]
    ].copy()
    timeline["reward"] = timeline["immediate_reward"].apply(
        lambda x: f"{x:.1f}" if pd.notna(x) else "-"
    )

    def highlight_negative(row: pd.Series) -> list[str]:
        val = row.get("immediate_reward")
        if pd.notna(val) and float(val) < 0:
            return ["background-color: rgba(231, 76, 60, 0.25)"] * len(row)
        return [""] * len(row)

    st.markdown("**Decision timeline**")
    st.dataframe(
        timeline.drop(columns=["immediate_reward"]).style.apply(highlight_negative, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="STS2 Agent Dashboard",
        page_icon="🃏",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    revision = _data_revision_key()
    runs_raw = load_runs(revision)
    decisions_raw = load_decisions(revision)
    card_choices_raw = load_card_choices(revision)
    runs_raw = enrich_runs_with_duration(runs_raw, decisions_raw)
    sidebar = render_sidebar(runs_raw)

    view_mode = sidebar.get("view_mode", VIEW_COMPARE)
    runs, decisions, card_choices = apply_filters(
        runs_raw, decisions_raw, card_choices_raw, sidebar
    )

    card_filter_parts: list[str] = []
    if view_mode in (VIEW_HUMAN, VIEW_COMPARE) and not card_choices.empty:
        card_filter_parts.append(f"{len(card_choices)} human reward picks")
    if view_mode in (VIEW_AGENT, VIEW_COMPARE, VIEW_COMPARE_VERSIONS) and not decisions.empty:
        card_filter_parts.append(f"{len(decisions)} agent decisions")
    if sidebar.get("character") and sidebar["character"] != "All":
        card_filter_parts.append(sidebar["character"])
    card_filter_label = " · ".join(card_filter_parts) if card_filter_parts else None

    tab_runs, tab_compendium = st.tabs(["Run analytics", "Enemy compendium"])

    with tab_runs:
        section_metrics(runs, runs_raw, view_mode, sidebar)
        st.divider()
        section_run_timing(runs, view_mode, sidebar)
        st.divider()
        section_win_rate(runs, view_mode, sidebar)
        st.divider()
        section_death(runs, view_mode, sidebar)
        st.divider()
        section_act_progression(runs, view_mode, sidebar)
        st.divider()
        section_combat(runs, decisions, view_mode, sidebar)
        st.divider()
        section_cards(
            decisions,
            card_choices,
            runs,
            view_mode,
            filter_label=card_filter_label,
        )
        st.divider()
        section_deck_stats(runs, view_mode, sidebar)
        st.divider()
        section_recent_runs(runs, view_mode, sidebar)
        st.divider()
        section_decision_explorer(runs, decisions, card_choices)

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
