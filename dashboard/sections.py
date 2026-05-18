"""Phase A dashboard sections (four analytics blocks)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import metrics as m

PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 380
FILTER_DETAIL_VERSION = "dash_detail_agent_version"


def _empty_chart(title: str, message: str = "No data yet") -> go.Figure:
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


def _trend_chart(df: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    if df.empty:
        return _empty_chart(title)
    fig = px.line(
        df,
        x="version_run_number",
        y="value",
        color="version",
        markers=True,
        title=title,
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        height=CHART_HEIGHT,
        xaxis_title="Run # (per version)",
        yaxis_title=y_title,
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def render_phase_b_note(clean_count: int, agent_total: int) -> bool:
    """Show Phase B run count; return False when there is nothing to analyze."""
    if clean_count <= 0:
        st.caption("No runs with Phase B logging (`combat_summary`) for this version and filters.")
        return False
    excluded = agent_total - clean_count
    suffix = f" ({excluded} older runs excluded)" if excluded > 0 else ""
    st.caption(f"Based on **{clean_count}** runs with Phase B logging{suffix}.")
    return True


def render_detail_version_picker(
    versions: list[str],
    default: str | None,
) -> str | None:
    """Single-version selector for sections 2–4 (defaults to newest agent version)."""
    if not versions:
        return None
    if (
        FILTER_DETAIL_VERSION not in st.session_state
        or st.session_state[FILTER_DETAIL_VERSION] not in versions
    ):
        pick = default if default in versions else versions[-1]
        st.session_state[FILTER_DETAIL_VERSION] = pick

    st.subheader("Detailed analysis")
    st.caption(
        "Sections 2–4 below use one agent version. "
        "Section 1 uses **Versions for overview** in the sidebar."
    )
    return st.selectbox(
        "Agent version to analyze",
        versions,
        key=FILTER_DETAIL_VERSION,
    )


def section_health(runs: pd.DataFrame) -> None:
    st.header("1. Health at a glance")
    st.caption("Uses **Versions for overview** from the sidebar (all checked versions).")
    gv = m.game_version_caption(runs)
    if gv:
        st.caption(gv)

    summary = m.summary_row_per_version(runs)
    if summary.empty:
        st.info("No agent runs for these filters.")
        return

    st.dataframe(summary.drop(columns=[c for c in summary.columns if c.startswith("_")], errors="ignore"), hide_index=True)

    warnings = m.early_version_warnings(runs)
    if warnings:
        st.warning("Early warning (≥15% lower avg floor at same run count):\n\n" + "\n".join(f"- {w}" for w in warnings))

    floor_trend, score_trend = m.rolling_trend_by_version(runs)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_trend_chart(floor_trend, "Rolling avg floor (last 50 runs)", "Floor"), use_container_width=True)
    with c2:
        if not score_trend.empty:
            st.plotly_chart(_trend_chart(score_trend, "Rolling avg score (last 50 runs)", "Score"), use_container_width=True)
        else:
            st.plotly_chart(_empty_chart("Rolling avg score", "No run_score in dataset"), use_container_width=True)


def section_death(
    runs: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    agent_version: str | None = None,
    phase_b_count: int = 0,
    phase_b_total: int = 0,
) -> None:
    st.header("2. Where and why it dies")
    if agent_version:
        st.caption(f"Agent version: **{agent_version}**")
    if not render_phase_b_note(phase_b_count, phase_b_total):
        return
    agent = m.agent_runs_only(runs)
    if agent.empty:
        st.info("No agent runs for these filters.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Floor reached (losses)**")
        losses = agent[agent["won"] == False]  # noqa: E712
        if losses.empty:
            st.caption("No losses yet.")
        else:
            fig = px.histogram(
                losses,
                x="floors_reached",
                nbins=min(20, max(5, int(losses["floors_reached"].max() or 5))),
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(height=CHART_HEIGHT, xaxis_title="Floor", yaxis_title="Runs")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Death category**")
        if "cause_of_death" in losses.columns and not losses.empty:
            cats = losses["cause_of_death"].apply(m.parse_death_category)
            cat_df = cats.value_counts().reset_index()
            cat_df.columns = ["category", "count"]
            fig = px.bar(cat_df, x="category", y="count", template=PLOTLY_TEMPLATE)
            fig.update_layout(height=CHART_HEIGHT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No cause_of_death data.")

    enemy_df = m.death_enemy_counts(runs)
    if not enemy_df.empty:
        st.markdown("**Enemy killer** (parsed from cause_of_death)")
        fig = px.bar(enemy_df, x="enemy", y="deaths", template=PLOTLY_TEMPLATE)
        fig.update_layout(height=280, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Act reached (all runs)**")
    act_df = agent["act_reached"].value_counts().sort_index().reset_index()
    act_df.columns = ["act", "runs"]
    fig = px.bar(act_df, x="act", y="runs", template=PLOTLY_TEMPLATE)
    fig.update_layout(height=260)
    st.plotly_chart(fig, use_container_width=True)

    hp_act = m.hp_at_act_entry(decisions)
    if not hp_act.empty:
        st.markdown("**HP entering act** (from decision snapshots)")
        fig = px.box(hp_act, x="act", y="hp_entering_act", template=PLOTLY_TEMPLATE)
        fig.update_layout(height=CHART_HEIGHT)
        st.plotly_chart(fig, use_container_width=True)

    by_enemy = m.aggregate_combat_by_enemy(runs)
    if not by_enemy.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Avg turns per fight** (by enemy, `combat_summary`)")
            fig = px.bar(
                by_enemy,
                x="enemy",
                y="avg_turns",
                template=PLOTLY_TEMPLATE,
                hover_data=["fights"],
            )
            fig.update_layout(
                height=CHART_HEIGHT,
                xaxis_title="Enemy",
                yaxis_title="Avg turns",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Avg damage taken per fight** (by enemy, `combat_summary`)")
            fig = px.bar(
                by_enemy,
                x="enemy",
                y="avg_damage_taken",
                template=PLOTLY_TEMPLATE,
                hover_data=["fights"],
            )
            fig.update_layout(
                height=CHART_HEIGHT,
                xaxis_title="Enemy",
                yaxis_title="Avg damage taken",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig, use_container_width=True)

    human_dmg = m.human_benchmark_damage(runs)

    if human_dmg is not None and "total_damage_taken" in agent.columns:
        st.markdown("**Run-total damage vs human benchmark**")
        agent_mean = float(agent["total_damage_taken"].mean())
        col1, col2 = st.columns(2)
        col1.metric("Agent avg damage taken / run", f"{agent_mean:.0f}")
        col2.metric("Human benchmark (avg)", f"{human_dmg:.0f}")
        if agent_mean > human_dmg * 1.1:
            st.caption("Agent takes more damage than human benchmark on average.")


def section_cards(
    runs: pd.DataFrame,
    decisions: pd.DataFrame,
    card_choices: pd.DataFrame,
    *,
    agent_version: str | None = None,
    phase_b_count: int = 0,
    phase_b_total: int = 0,
) -> None:
    st.header("3. Card decisions")
    if agent_version:
        st.caption(f"Agent version: **{agent_version}**")
    if not render_phase_b_note(phase_b_count, phase_b_total):
        return
    agent_picks = m.agent_reward_picks(decisions)
    human_picks = m.human_reward_picks(card_choices)

    table = m.pick_rate_table(agent_picks, human_picks)
    if not table.empty:
        st.markdown("**Pick rates** (aggregate; matched-offer compare is Phase B)")
        st.dataframe(table, hide_index=True, use_container_width=True)
    else:
        st.caption("No card reward picks in filtered data.")

    tier_df = m.picked_card_tier_counts(agent_picks)
    if not tier_df.empty:
        st.markdown("**Agent pick tiers** (Mobalytics)")
        fig = px.bar(tier_df, x="tier", y="picks", template=PLOTLY_TEMPLATE)
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

    human_miss = m.human_tier_miss_rate(card_choices)
    if human_miss is not None:
        st.metric("Human tier-miss rate", f"{human_miss:.1f}%", help="Picked tier ≥1 letter worse than best offered")

    st.caption(
        "Agent tier-miss (“left S on table”) needs offered cards in logs — **Phase B**. "
        "Human tier-miss uses imported card_choices.jsonl."
    )

    deck_df = m.deck_card_win_loss(runs)
    if not deck_df.empty:
        st.markdown("**Cards in final decks** (wins vs losses)")
        st.dataframe(deck_df.head(12), hide_index=True, use_container_width=True)


def section_combat_efficiency(
    runs: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    agent_version: str | None = None,
    phase_b_count: int = 0,
    phase_b_total: int = 0,
) -> None:
    st.header("4. Combat efficiency")
    if agent_version:
        st.caption(f"Agent version: **{agent_version}**")
    if not render_phase_b_note(phase_b_count, phase_b_total):
        return
    block = m.block_efficiency(decisions)
    potions = m.potion_hoard_death_rate(runs)
    energy = m.energy_waste_rate(decisions)

    c1, c2, c3 = st.columns(3)
    if block is not None:
        c1.metric(
            "Block efficiency",
            f"{block:.1f}%",
            help="% combat turns with block gain when enemy incoming intent > 0",
        )
    else:
        c1.caption("Block efficiency: no combat intent data")

    if potions is not None:
        c2.metric(
            "Potion hoarding at death",
            f"{potions:.1f}%",
            help="% deaths with at least one empty potion slot",
        )
    else:
        c2.caption("Potion metric: no deaths")

    if energy is not None:
        c3.metric("Energy left on end_turn", f"{energy:.1f}%")
    else:
        c3.caption("Energy waste: no end_turn rows")

    by_enemy = m.aggregate_combat_by_enemy(runs)
    if not by_enemy.empty and by_enemy["avg_damage_dealt"].notna().any():
        st.markdown("**Avg damage dealt per fight** (by enemy, `combat_summary`)")
        fig = px.bar(
            by_enemy,
            x="enemy",
            y="avg_damage_dealt",
            template=PLOTLY_TEMPLATE,
            hover_data=["fights"],
        )
        fig.update_layout(
            height=CHART_HEIGHT,
            xaxis_title="Enemy",
            yaxis_title="Avg damage dealt",
            xaxis_tickangle=-35,
        )
        st.plotly_chart(fig, use_container_width=True)
