"""Analytics helpers for the STS2 Streamlit dashboard (Phase A)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

COMBAT_TYPES = frozenset({"monster", "elite", "boss", "hand_select"})

TIER_ORDER = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
TIER_LETTERS = ("S", "A", "B", "C", "D")


def agent_runs_only(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or "source" not in runs.columns:
        return runs
    return runs[runs["source"] != "human"].copy()


def human_runs_only(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or "source" not in runs.columns:
        return pd.DataFrame()
    return runs[runs["source"] == "human"].copy()


def run_has_combat_summary(value: object) -> bool:
    """True when run row has a non-empty combat_summary list (Phase B)."""
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return False


def runs_with_combat_summary(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    if "combat_summary" not in runs.columns:
        return runs.iloc[0:0]
    return runs[runs["combat_summary"].apply(run_has_combat_summary)].copy()


def filter_detail_phase_b(
    runs: pd.DataFrame,
    decisions: pd.DataFrame,
    card_choices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int]:
    """
    Sections 2–4: agent runs must have combat_summary; keep human runs for benchmarks.
    Returns (runs, decisions, card_choices, clean_agent_count, total_agent_before_filter).
    """
    agent = agent_runs_only(runs)
    total_agent = len(agent)
    clean_agent = runs_with_combat_summary(agent)
    clean_count = len(clean_agent)
    clean_ids = (
        set(clean_agent["run_id"].astype(str))
        if not clean_agent.empty and "run_id" in clean_agent.columns
        else set()
    )

    if runs.empty:
        return runs, decisions.iloc[0:0], card_choices, 0, total_agent

    if "source" in runs.columns:
        human = runs[runs["source"] == "human"]
        agent_part = runs[runs["source"] != "human"]
        if clean_ids:
            agent_part = agent_part[agent_part["run_id"].astype(str).isin(clean_ids)]
        else:
            agent_part = agent_part.iloc[0:0]
        runs_out = pd.concat([human, agent_part], ignore_index=True)
    else:
        runs_out = clean_agent

    if decisions.empty or "run_id" not in decisions.columns:
        dec_out = decisions.iloc[0:0]
    elif clean_ids:
        dec_out = decisions[decisions["run_id"].astype(str).isin(clean_ids)].copy()
    else:
        dec_out = decisions.iloc[0:0]

    return (
        runs_out.reset_index(drop=True),
        dec_out.reset_index(drop=True),
        card_choices,
        clean_count,
        total_agent,
    )


def version_groups(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    agent = agent_runs_only(runs)
    if agent.empty or "agent_version" not in agent.columns:
        return {}
    out: dict[str, pd.DataFrame] = {}
    for ver, grp in agent.groupby("agent_version", sort=False):
        ordered = grp.sort_values("timestamp").reset_index(drop=True)
        ordered = ordered.copy()
        ordered["version_run_number"] = range(1, len(ordered) + 1)
        out[str(ver)] = ordered
    return out


def game_version_caption(runs: pd.DataFrame) -> str:
    if runs.empty or "game_version" not in runs.columns:
        return ""
    versions = sorted({str(v) for v in runs["game_version"].dropna().unique() if str(v)})
    if not versions:
        return ""
    return "Game version(s): " + ", ".join(versions)


def summary_row_per_version(runs: pd.DataFrame) -> pd.DataFrame:
    groups = version_groups(runs)
    rows: list[dict[str, Any]] = []
    for ver, df in sorted(groups.items()):
        row: dict[str, Any] = {
            "Version": ver,
            "Runs": len(df),
            "Avg floor": round(float(df["floors_reached"].mean()), 1) if len(df) else None,
        }
        if "run_score" in df.columns and df["run_score"].sum() > 0:
            row["Avg score"] = round(float(df["run_score"].mean()), 0)
        else:
            row["Avg score"] = None
        if "run_duration_sec" in df.columns and df["run_duration_sec"].notna().any():
            mean_sec = float(df["run_duration_sec"].dropna().mean())
            row["Avg duration"] = format_duration(mean_sec)
        else:
            row["Avg duration"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_trend_by_version(
    runs: pd.DataFrame,
    *,
    window: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    floor_rows: list[dict] = []
    score_rows: list[dict] = []
    for ver, df in version_groups(runs).items():
        tail = df.tail(window)
        if tail.empty:
            continue
        floors = tail["floors_reached"].astype(float)
        floor_roll = floors.rolling(window=min(window, len(tail)), min_periods=1).mean()
        for i, (_, row) in enumerate(tail.iterrows()):
            floor_rows.append(
                {
                    "version": ver,
                    "version_run_number": int(row["version_run_number"]),
                    "value": float(floor_roll.iloc[i]),
                }
            )
        if "run_score" in tail.columns and tail["run_score"].sum() > 0:
            scores = tail["run_score"].astype(float)
            score_roll = scores.rolling(window=min(window, len(tail)), min_periods=1).mean()
            for i, (_, row) in enumerate(tail.iterrows()):
                score_rows.append(
                    {
                        "version": ver,
                        "version_run_number": int(row["version_run_number"]),
                        "value": float(score_roll.iloc[i]),
                    }
                )
    return pd.DataFrame(floor_rows), pd.DataFrame(score_rows)


def early_version_warnings(
    runs: pd.DataFrame,
    *,
    window: int = 50,
    threshold: float = 0.15,
) -> list[str]:
    groups = version_groups(runs)
    versions = sorted(groups.keys())
    warnings: list[str] = []

    def avg_floor_at_n(ver: str, n: int) -> float | None:
        df = groups[ver]
        if df.empty:
            return None
        n = min(n, len(df))
        return float(df.head(n)["floors_reached"].mean())

    for i, ver_a in enumerate(versions):
        for ver_b in versions[i + 1 :]:
            n = min(len(groups[ver_a]), len(groups[ver_b]), window)
            if n < 5:
                continue
            fa = avg_floor_at_n(ver_a, n)
            fb = avg_floor_at_n(ver_b, n)
            if fa is None or fb is None or fb <= 0:
                continue
            if fa < fb * (1.0 - threshold):
                pct = (1.0 - fa / fb) * 100
                warnings.append(
                    f"**{ver_a}** avg floor **{fa:.1f}** vs **{ver_b}** **{fb:.1f}** "
                    f"at first **{n}** runs (−{pct:.0f}%)"
                )
            if fb < fa * (1.0 - threshold):
                pct = (1.0 - fb / fa) * 100
                warnings.append(
                    f"**{ver_b}** avg floor **{fb:.1f}** vs **{ver_a}** **{fa:.1f}** "
                    f"at first **{n}** runs (−{pct:.0f}%)"
                )
    return warnings


def parse_death_enemy(cause: str | None) -> str | None:
    if not cause or not isinstance(cause, str):
        return None
    match = re.search(r"vs\s+(.+?)\s+-\s+hp reached 0", cause, re.I)
    if match:
        return match.group(1).strip()
    return None


def parse_death_category(cause: str | None) -> str:
    if not cause or not isinstance(cause, str):
        return "Unknown"
    imported = parse_encounter_death_category(cause)
    if imported:
        return imported
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
    if lower.startswith("game_over"):
        return "Unknown"
    return "Other"


def parse_encounter_death_category(cause: str) -> str | None:
    """Map imported game ids (ENCOUNTER.* / EVENT.*) to room type."""
    if not cause or not isinstance(cause, str):
        return None
    upper = cause.strip().upper()
    if upper.startswith("EVENT."):
        return "Event"
    if "_BOSS" in upper or upper.endswith("BOSS"):
        return "Boss"
    if "_ELITE" in upper or ".ELITE" in upper:
        return "Elite"
    if upper.startswith("ENCOUNTER."):
        return "Monster"
    return None


def parse_encounter_death_enemy(cause: str) -> str | None:
    """Human-readable label from ENCOUNTER.SLUG_NORMAL style ids."""
    if not cause or not isinstance(cause, str):
        return None
    upper = cause.strip().upper()
    if not upper.startswith(("ENCOUNTER.", "EVENT.")):
        return None
    slug = cause.split(".", 1)[-1]
    for suffix in ("_NORMAL", "_WEAK", "_ELITE", "_BOSS", "_HARD"):
        if slug.upper().endswith(suffix):
            slug = slug[: -len(suffix)]
            break
    slug = slug.strip("_")
    if not slug:
        return None
    return slug.replace("_", " ").title()


def last_fatal_combat_fight(run: dict[str, Any]) -> dict[str, Any] | None:
    """Last lost fight in combat_summary; on losses this is usually the fatal fight."""
    summary = run.get("combat_summary")
    if not isinstance(summary, list) or not summary:
        return None
    for fight in reversed(summary):
        if isinstance(fight, dict) and fight.get("won_fight") is False:
            return fight
    last = summary[-1]
    return last if isinstance(last, dict) else None


def category_from_combat_state_type(state_type: object) -> str | None:
    st = str(state_type or "").lower()
    if st == "monster":
        return "Monster"
    if st == "elite":
        return "Elite"
    if st == "boss":
        return "Boss"
    return None


def resolve_death_category(run: dict[str, Any]) -> str:
    """Room type where the run ended (monster / elite / boss / …)."""
    fight = last_fatal_combat_fight(run)
    if fight:
        cat = category_from_combat_state_type(fight.get("state_type"))
        if cat:
            return cat

    killing = run.get("killing_enemy")
    if isinstance(killing, dict) and killing.get("name"):
        cause = run.get("cause_of_death")
        if isinstance(cause, str):
            cat = parse_death_category(cause)
            if cat not in ("Unknown", "Other"):
                return cat

    cause = run.get("cause_of_death")
    if isinstance(cause, str) and cause.strip():
        return parse_death_category(cause)

    return "Unknown"


def resolve_death_enemy(run: dict[str, Any]) -> str | None:
    """Enemy label for a loss (single fight or encounter group)."""
    killing = run.get("killing_enemy")
    if isinstance(killing, dict):
        name = str(killing.get("name") or "").strip()
        if name:
            return name

    fight = last_fatal_combat_fight(run)
    if fight:
        label = format_enemy_label(fight.get("enemy_names"))
        if label != "Unknown":
            return label

    cause = run.get("cause_of_death")
    if isinstance(cause, str) and cause.strip():
        enemy = parse_death_enemy(cause)
        if enemy:
            return enemy
        return parse_encounter_death_enemy(cause)

    return None


def format_enemy_label(enemy_names: object) -> str:
    """Stable label for a fight from combat_summary.enemy_names."""
    if not isinstance(enemy_names, list) or not enemy_names:
        return "Unknown"
    names = sorted({str(n).strip() for n in enemy_names if n and str(n).strip()})
    return ", ".join(names) if names else "Unknown"


def combat_summary_fights(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per fight from runs.combat_summary (Phase B agent runs)."""
    rows: list[dict] = []
    for _, run in agent_runs_only(runs).iterrows():
        summary = run.get("combat_summary") or []
        if not isinstance(summary, list):
            continue
        for fight in summary:
            if not isinstance(fight, dict):
                continue
            won = fight.get("won_fight")
            rows.append(
                {
                    "run_id": run.get("run_id"),
                    "enemy": format_enemy_label(fight.get("enemy_names")),
                    "turns": fight.get("turns"),
                    "damage_taken": fight.get("damage_taken"),
                    "damage_dealt": fight.get("damage_dealt"),
                    "state_type": fight.get("state_type"),
                    "won_fight": bool(won) if won is not None else None,
                }
            )
    return pd.DataFrame(rows)


def enemy_fight_win_rates(
    runs: pd.DataFrame,
    *,
    min_fights: int = 3,
    top_n: int = 20,
) -> pd.DataFrame:
    """Per-enemy fight win % from combat_summary (won_fight), grouped by encounter label."""
    fights = combat_summary_fights(runs)
    if fights.empty or "won_fight" not in fights.columns:
        return pd.DataFrame()

    fights = fights[fights["enemy"] != "Unknown"].copy()
    fights = fights[fights["won_fight"].notna()]
    if fights.empty:
        return pd.DataFrame()

    agg = (
        fights.groupby("enemy", as_index=False)
        .agg(fights=("enemy", "count"), wins=("won_fight", "sum"))
        .astype({"wins": int})
    )
    agg["win_rate"] = (100.0 * agg["wins"] / agg["fights"]).round(1)
    agg = agg[agg["fights"] >= max(1, min_fights)].sort_values(
        ["win_rate", "fights"], ascending=[True, False]
    )
    if top_n > 0:
        agg = agg.head(top_n)
    return agg.reset_index(drop=True)


def aggregate_combat_by_enemy(runs: pd.DataFrame) -> pd.DataFrame:
    """Average turns and damage per enemy label across all Phase B fights."""
    fights = combat_summary_fights(runs)
    if fights.empty:
        return pd.DataFrame()

    for col in ("turns", "damage_taken", "damage_dealt"):
        if col in fights.columns:
            fights[col] = pd.to_numeric(fights[col], errors="coerce")

    agg = (
        fights.groupby("enemy", as_index=False)
        .agg(
            fights=("enemy", "count"),
            avg_turns=("turns", "mean"),
            avg_damage_taken=("damage_taken", "mean"),
            avg_damage_dealt=("damage_dealt", "mean"),
        )
        .sort_values("fights", ascending=False)
    )
    agg["avg_turns"] = agg["avg_turns"].round(1)
    agg["avg_damage_taken"] = agg["avg_damage_taken"].round(1)
    agg["avg_damage_dealt"] = agg["avg_damage_dealt"].round(1)
    return agg


def per_fight_hp_damage(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, run in runs.iterrows():
        before = run.get("hp_before_each_combat") or []
        after = run.get("hp_after_each_combat") or []
        if not isinstance(before, list) or not isinstance(after, list):
            continue
        n = min(len(before), len(after))
        for i in range(n):
            b, a = int(before[i]), int(after[i])
            rows.append(
                {
                    "run_id": run.get("run_id"),
                    "agent_version": run.get("agent_version"),
                    "source": run.get("source"),
                    "fight_index": i + 1,
                    "hp_start": b,
                    "hp_end": a,
                    "damage_taken": max(0, b - a),
                }
            )
    return pd.DataFrame(rows)


def combat_sessions_from_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for run_id, grp in decisions.groupby("run_id", sort=False):
        grp = grp.sort_values("timestamp")
        session: list[pd.Series] = []
        in_combat = False

        def flush() -> None:
            if not session:
                return
            first, last = session[0], session[-1]
            hp_start = _snap_hp(first)
            hp_end = _snap_hp(last)
            rows.append(
                {
                    "run_id": run_id,
                    "agent_version": first.get("agent_version"),
                    "fight_type": str(first.get("state_type") or "combat"),
                    "decisions_in_fight": len(session),
                    "hp_start": hp_start,
                    "hp_end": hp_end,
                    "damage_taken": max(0, (hp_start or 0) - (hp_end or 0))
                    if hp_start is not None and hp_end is not None
                    else None,
                    "damage_dealt": _sum_damage_dealt(session),
                }
            )

        for _, row in grp.iterrows():
            st = str(row.get("state_type") or "")
            if st in COMBAT_TYPES:
                if not in_combat:
                    in_combat = True
                    session = []
                session.append(row)
            else:
                if in_combat:
                    flush()
                    in_combat = False
                    session = []
        if in_combat:
            flush()

    return pd.DataFrame(rows)


def _snap_hp(row: pd.Series) -> int | None:
    val = row.get("player_hp")
    if pd.notna(val):
        try:
            return int(val)
        except (TypeError, ValueError):
            pass
    return None


def _sum_damage_dealt(session: list[pd.Series]) -> int:
    total = 0
    for row in session:
        dealt = row.get("damage_dealt_turn")
        if pd.notna(dealt):
            total += int(dealt)
            continue
        reward = row.get("immediate_reward")
        if isinstance(reward, dict):
            total += int(reward.get("damage_dealt") or 0)
    return total


def hp_at_act_entry(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty or "act" not in decisions.columns:
        return pd.DataFrame()
    rows: list[dict] = []
    for (run_id, act), grp in decisions.groupby(["run_id", "act"], sort=False):
        grp = grp.sort_values("timestamp")
        hp = _snap_hp(grp.iloc[0])
        if hp is None:
            continue
        rows.append(
            {
                "run_id": run_id,
                "act": int(act),
                "hp_entering_act": hp,
                "agent_version": grp.iloc[0].get("agent_version"),
            }
        )
    return pd.DataFrame(rows)


def human_benchmark_damage(runs: pd.DataFrame) -> float | None:
    human = human_runs_only(runs)
    if human.empty or "total_damage_taken" not in human.columns:
        return None
    return float(human["total_damage_taken"].mean())


def normalize_card_name(name: str | None) -> str | None:
    """
    Canonical card key for matching agent vs human (UPPER_UNDERSCORE).
    Accepts display names, ids, or CARD.* prefixes.
    """
    if name is None:
        return None
    text = str(name).strip()
    if not text or text.lower().startswith("reward slot"):
        return None
    if text.upper().startswith("CARD."):
        text = text.split(".", 1)[-1].strip()
    key = text.upper().replace(" ", "_").replace("-", "_")
    key = re.sub(r"_+", "_", key).strip("_")
    return key or None


def format_card_label(canonical: str) -> str:
    """Readable label for dashboard tables."""
    return str(canonical).replace("_", " ").title()


def normalize_pick_list(picks: list[str]) -> list[str]:
    out: list[str] = []
    for pick in picks:
        key = normalize_card_name(pick)
        if key:
            out.append(key)
    return out


def _card_index_from_row(row: pd.Series) -> int | None:
    idx = row.get("card_index")
    if pd.notna(idx):
        try:
            return int(idx)
        except (TypeError, ValueError):
            pass
    text = str(row.get("action_reasoning") or "")
    match = re.search(r"key=select_card_reward:(\d+)", text)
    if match:
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            pass
    return None


def _name_from_offered_list(offered: list, card_index: int) -> str | None:
    """Best-effort: map API index to offered name list (usually same order)."""
    if not offered or card_index < 0:
        return None
    if card_index < len(offered):
        return str(offered[card_index])
    return None


def extract_card_pick_name(row: pd.Series) -> str | None:
    picked = row.get("card_reward_picked")
    if picked and str(picked).strip():
        return str(picked).strip()

    text = str(row.get("action_reasoning") or "")
    match = re.search(r"best card:\s*([^(;]+)", text, re.I)
    if match:
        return match.group(1).strip()

    card_index = _card_index_from_row(row)
    offered = row.get("card_reward_offered")
    if isinstance(offered, list) and card_index is not None:
        name = _name_from_offered_list(offered, card_index)
        if name:
            return name

    if card_index is not None:
        return f"reward slot {card_index}"
    return None


def agent_reward_picks(decisions: pd.DataFrame) -> list[str]:
    if decisions.empty:
        return []
    mask = (decisions["state_type"] == "card_reward") & (
        decisions["action"].isin(["select_card_reward", "select_card"])
    )
    raw = (
        decisions.loc[mask]
        .apply(extract_card_pick_name, axis=1)
        .dropna()
        .astype(str)
        .tolist()
    )
    return normalize_pick_list(raw)


def human_reward_picks(card_choices: pd.DataFrame) -> list[str]:
    if card_choices.empty or "picked" not in card_choices.columns:
        return []
    return normalize_pick_list(card_choices["picked"].dropna().astype(str).tolist())


def pick_rate_table(
    agent_picks: list[str],
    human_picks: list[str],
    *,
    top_n: int = 12,
) -> pd.DataFrame:
    ac = Counter(normalize_pick_list(agent_picks))
    hc = Counter(normalize_pick_list(human_picks))
    cards = {c for c, _ in ac.most_common(top_n)} | {c for c, _ in hc.most_common(top_n)}
    if not cards:
        return pd.DataFrame()
    a_total = max(len(agent_picks), 1)
    h_total = max(len(human_picks), 1)
    rows = []
    for card in sorted(cards, key=lambda c: ac.get(c, 0) + hc.get(c, 0), reverse=True):
        rows.append(
            {
                "Card": format_card_label(card),
                "Agent picks": ac.get(card, 0),
                "Agent %": f"{100 * ac.get(card, 0) / a_total:.1f}",
                "Human picks": hc.get(card, 0),
                "Human %": f"{100 * hc.get(card, 0) / h_total:.1f}",
            }
        )
    return pd.DataFrame(rows)


def tier_rank(tier: str | None) -> int:
    if not tier:
        return -1
    letter = str(tier).strip().upper()[:1]
    return TIER_ORDER.get(letter, -1)


def card_tier(name: str) -> str | None:
    from sts2_agent.knowledge import get_knowledge

    return get_knowledge().expert_card_tier(name)


def picked_card_tier_counts(pick_names: list[str]) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for name in normalize_pick_list(pick_names):
        tier = card_tier(name) or "?"
        counts[tier] += 1
    if not counts:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"tier": t, "picks": counts[t]} for t in sorted(counts.keys(), key=lambda x: tier_rank(x), reverse=True)]
    )


def human_tier_miss_rate(card_choices: pd.DataFrame) -> float | None:
    """% of human picks where picked tier is 1+ letter worse than best offered."""
    if card_choices.empty or "offered" not in card_choices.columns:
        return None
    misses = 0
    total = 0
    for _, row in card_choices.iterrows():
        offered = row.get("offered") or []
        picked = normalize_card_name(row.get("picked"))
        if not isinstance(offered, list) or not picked:
            continue
        tiers = []
        for card in offered:
            key = normalize_card_name(str(card))
            if not key:
                continue
            tr = tier_rank(card_tier(key))
            if tr >= 0:
                tiers.append(tr)
        if not tiers:
            continue
        best = max(tiers)
        picked_t = tier_rank(card_tier(picked))
        if picked_t < 0:
            continue
        total += 1
        if picked_t <= best - 1:
            misses += 1
    if total == 0:
        return None
    return 100.0 * misses / total


def deck_card_win_loss(runs: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    wins = runs[runs["won"] == True]  # noqa: E712
    losses = runs[runs["won"] == False]  # noqa: E712

    def count_decks(sub: pd.DataFrame) -> Counter:
        c: Counter = Counter()
        for deck in sub.get("final_deck", []):
            if isinstance(deck, list):
                for card in deck:
                    key = normalize_card_name(str(card))
                    if key:
                        c[key] += 1
        return c

    wc, lc = count_decks(wins), count_decks(losses)
    cards = {c for c, _ in wc.most_common(top_n)} | {c for c, _ in lc.most_common(top_n)}
    rows = []
    for card in sorted(cards, key=lambda c: wc.get(c, 0) + lc.get(c, 0), reverse=True):
        rows.append(
            {
                "Card": format_card_label(card),
                "In wins": wc.get(card, 0),
                "In losses": lc.get(card, 0),
            }
        )
    return pd.DataFrame(rows)


def parse_intent_damage_value(value: object) -> int:
    """Parse enemy intent damage: int, numeric string, or multi-hit 'NxM' (e.g. 4x2 -> 8)."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return max(0, int(value))

    text = str(value).strip().lower()
    if not text:
        return 0

    match = re.match(r"^(\d+)\s*x\s*(\d+)$", text)
    if match:
        return int(match.group(1)) * int(match.group(2))

    try:
        return max(0, int(float(text)))
    except (TypeError, ValueError):
        return 0


def _hp_lost_from_row(row: pd.Series) -> int:
    val = row.get("hp_lost_this_turn")
    if pd.notna(val):
        try:
            return max(0, int(val))
        except (TypeError, ValueError):
            pass
    reward = row.get("immediate_reward")
    if isinstance(reward, dict):
        try:
            return max(0, int(reward.get("hp_lost_this_turn") or 0))
        except (TypeError, ValueError):
            return 0
    return 0


def damage_mitigation_rate(decisions: pd.DataFrame) -> float | None:
    """
    Of incoming attack damage at end_turn, what % was absorbed by block?
    Per decision: min(1, max(0, incoming - hp_lost) / incoming); averaged * 100.
    """
    if decisions.empty or "incoming_damage" not in decisions.columns:
        return None
    if "action" not in decisions.columns:
        return None

    ends = decisions[decisions["action"] == "end_turn"]
    if ends.empty:
        return None

    mitigations: list[float] = []
    for _, row in ends.iterrows():
        try:
            incoming = int(row.get("incoming_damage") or 0)
        except (TypeError, ValueError):
            continue
        if incoming <= 0:
            continue
        hp_lost = _hp_lost_from_row(row)
        damage_mitigated = max(0, incoming - hp_lost)
        mitigations.append(min(1.0, damage_mitigated / incoming))

    if not mitigations:
        return None
    return 100.0 * sum(mitigations) / len(mitigations)


def _potion_belt_filled_count(row: pd.Series, *, default_max_slots: int = 3) -> tuple[int, int] | None:
    """
    Return (max_slots, filled_count) for a run row.
    New logs: len(potions_at_death) == max_potion_slots with None for empty slots.
    Legacy logs: filled-only list + optional max_potion_slots.
    """
    belt = row.get("potions_at_death")
    if not isinstance(belt, list):
        return None

    max_slots = row.get("max_potion_slots")
    if pd.notna(max_slots) and int(max_slots) > 0:
        cap = int(max_slots)
    else:
        cap = default_max_slots

    if len(belt) == cap:
        filled = sum(1 for p in belt if p)
        return cap, filled

    filled = sum(1 for p in belt if p)
    return cap, filled


def potion_hoard_death_rate(runs: pd.DataFrame, *, default_max_slots: int = 3) -> float | None:
    """% of deaths where the belt still had at least one unused potion."""
    agent = agent_runs_only(runs)
    deaths = agent[agent["won"] == False]  # noqa: E712
    if deaths.empty:
        return None
    hoard = 0
    counted = 0
    for _, row in deaths.iterrows():
        parsed = _potion_belt_filled_count(row, default_max_slots=default_max_slots)
        if parsed is None:
            continue
        max_slots, filled = parsed
        if max_slots <= 0:
            continue
        counted += 1
        if filled >= 1:
            hoard += 1
    if counted == 0:
        return None
    return 100.0 * hoard / counted


def energy_waste_rate(decisions: pd.DataFrame) -> float | None:
    """% of end_turn actions with energy still available."""
    if decisions.empty:
        return None
    ends = decisions[decisions["action"] == "end_turn"]
    if ends.empty:
        return None
    wasted = 0
    total = 0
    for _, row in ends.iterrows():
        energy = row.get("player_energy")
        if pd.isna(energy):
            continue
        total += 1
        if int(energy) > 0:
            wasted += 1
    if total == 0:
        return None
    return 100.0 * wasted / total


def death_category_counts(runs: pd.DataFrame) -> pd.DataFrame:
    """Death room-type counts for agent losses (combat_summary-first)."""
    agent = agent_runs_only(runs)
    deaths = agent[agent["won"] == False]  # noqa: E712
    if deaths.empty:
        return pd.DataFrame()
    counter: Counter[str] = Counter()
    for _, row in deaths.iterrows():
        counter[resolve_death_category(row.to_dict())] += 1
    if not counter:
        return pd.DataFrame()
    order = ["Monster", "Elite", "Boss", "Event", "Shop", "Interrupted", "Other", "Unknown"]
    items = sorted(counter.items(), key=lambda kv: (-kv[1], order.index(kv[0]) if kv[0] in order else 99))
    return pd.DataFrame(items, columns=["category", "count"])


def death_enemy_counts(runs: pd.DataFrame, *, top_n: int = 15) -> pd.DataFrame:
    """Top enemies on agent losses (killing_enemy → combat_summary → cause_of_death)."""
    agent = agent_runs_only(runs)
    deaths = agent[agent["won"] == False]  # noqa: E712
    if deaths.empty:
        return pd.DataFrame()
    counter: Counter[str] = Counter()
    for _, row in deaths.iterrows():
        enemy = resolve_death_enemy(row.to_dict())
        if enemy:
            counter[enemy] += 1
    if not counter:
        return pd.DataFrame()
    return pd.DataFrame(counter.most_common(top_n), columns=["enemy", "deaths"])


def incoming_damage_from_snapshot(snap: dict) -> int:
    total = 0
    for enemy in snap.get("enemies") or []:
        if not isinstance(enemy, dict):
            continue
        total += parse_intent_damage_value(enemy.get("intent_value"))
    return total


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0 or (isinstance(seconds, float) and np.isnan(seconds)):
        return "-"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"
