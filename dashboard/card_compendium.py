"""Dashboard — browse Spire Codex cards from data/cards/ by color section."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sts2_agent.codex_cards import (  # noqa: E402
    CARD_SECTIONS,
    CARDS_ROOT,
    SECTION_LABELS,
    load_all_cards,
    load_card_index,
)

FILTER_CARD_COLORS = "dash_filter_card_colors"
FILTER_CARD_SEARCH = "dash_filter_card_search"
FILTER_CARD_PICK = "dash_filter_card_pick"


@st.cache_data(ttl=300)
def _cached_index() -> dict:
    return load_card_index()


@st.cache_data(ttl=300)
def _cached_all_cards() -> list[dict]:
    return load_all_cards()


def _card_summary_row(card: dict) -> dict:
    return {
        "name": card.get("name"),
        "id": card.get("id"),
        "section": card.get("section") or card.get("color"),
        "type": card.get("type_key") or card.get("type"),
        "rarity": card.get("rarity_key") or card.get("rarity"),
        "cost": card.get("cost"),
        "damage": card.get("damage"),
        "block": card.get("block"),
    }


def _ensure_pick(key: str, options: list[str], default: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = default
    elif options and st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]


def render_card_compendium() -> None:
    st.header("Card compendium")
    st.caption(
        f"Spire Codex snapshot by color. Source: `{CARDS_ROOT.relative_to(PROJECT_ROOT)}`"
    )

    if st.button("Reload cards", key="reload_card_compendium"):
        st.cache_data.clear()
        st.rerun()

    index = _cached_index()
    if not index:
        st.warning(
            "No local card database found. Import from Spire Codex first:"
        )
        st.code("py tools/import_codex_cards.py --force", language="powershell")
        return

    all_cards = _cached_all_cards()
    if not all_cards:
        st.info("Index exists but no card files loaded.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cards", int(index.get("total_cards") or len(all_cards)))
    c2.metric("Sections", len(index.get("sections") or {}))
    c3.metric("Fetched (UTC)", str(index.get("fetched_at") or "")[:19])
    c4.metric("Unmapped", int(index.get("unmapped_count") or 0))

    section_keys = [k for k, _ in CARD_SECTIONS]
    section_labels = {k: SECTION_LABELS[k] for k in section_keys}
    counts = {
        k: int((index.get("sections") or {}).get(k, {}).get("count") or 0)
        for k in section_keys
    }

    with st.expander("Cards per section", expanded=False):
        overview = pd.DataFrame(
            [
                {"section": section_labels[k], "key": k, "count": counts.get(k, 0)}
                for k in section_keys
            ]
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)

    default_colors = section_keys
    if FILTER_CARD_COLORS not in st.session_state:
        st.session_state[FILTER_CARD_COLORS] = default_colors

    color_filter = st.multiselect(
        "Color / pool filter",
        options=section_keys,
        format_func=lambda k: f"{section_labels[k]} ({counts.get(k, 0)})",
        key=FILTER_CARD_COLORS,
    )
    if not color_filter:
        st.info("Select at least one color section to browse.")
        return

    search = st.text_input(
        "Search (name or id)",
        placeholder="e.g. Bash, STRIKE, vulnerable",
        key=FILTER_CARD_SEARCH,
    ).strip().lower()

    filtered: list[dict] = []
    for card in all_cards:
        section = str(card.get("section") or card.get("color") or "").lower()
        if section not in color_filter:
            continue
        if search:
            blob = " ".join(
                str(card.get(k) or "")
                for k in ("name", "id", "description", "description_raw", "type", "type_key")
            ).lower()
            if search not in blob:
                continue
        filtered.append(card)

    st.caption(f"Showing **{len(filtered)}** of {len(all_cards)} cards")

    if not filtered:
        st.info("No cards match filters.")
        return

    table = pd.DataFrame(_card_summary_row(c) for c in filtered)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "cost": st.column_config.NumberColumn(format="%d"),
            "damage": st.column_config.NumberColumn(format="%d"),
            "block": st.column_config.NumberColumn(format="%d"),
        },
    )

    pick_labels = [
        f"{c.get('name')} [{c.get('id')}] ({section_labels.get(str(c.get('section') or ''), c.get('section'))})"
        for c in filtered
    ]
    pick_ids = [str(c.get("id") or c.get("name")) for c in filtered]

    default_id = pick_ids[0]
    if FILTER_CARD_PICK in st.session_state:
        prev = st.session_state[FILTER_CARD_PICK]
        if prev in pick_ids:
            default_id = prev

    _ensure_pick(FILTER_CARD_PICK, pick_ids, default_id)
    selected_id = st.selectbox(
        "Open card (exact Codex JSON)",
        pick_ids,
        format_func=lambda cid: pick_labels[pick_ids.index(cid)],
        key=FILTER_CARD_PICK,
    )

    selected = next(c for c in filtered if str(c.get("id") or c.get("name")) == selected_id)
    section = str(selected.get("section") or selected.get("color") or "")

    st.subheader(str(selected.get("name") or selected_id))
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ID", str(selected.get("id") or "—"))
    m2.metric("Type", str(selected.get("type_key") or selected.get("type") or "—"))
    m3.metric("Rarity", str(selected.get("rarity_key") or selected.get("rarity") or "—"))
    m4.metric("Cost", selected.get("cost") if selected.get("cost") is not None else "—")
    m5.metric("Section", section_labels.get(section, section))

    desc = selected.get("description") or selected.get("description_raw")
    if desc:
        st.markdown(str(desc))

    if selected.get("upgrade"):
        with st.expander("Upgrade variant"):
            st.json(selected["upgrade"])

    st.subheader("Exact JSON entry")
    st.caption("Full object stored under `data/cards/by_color/<section>.json` — use for sim / agent work.")
    st.json(selected)

    with st.expander("Copy as formatted JSON"):
        st.code(json.dumps(selected, ensure_ascii=False, indent=2), language="json")

    with st.expander("Section file metadata"):
        section_path = CARDS_ROOT / "by_color" / f"{section}.json"
        if section_path.exists():
            meta = json.loads(section_path.read_text(encoding="utf-8"))
            st.json(
                {
                    k: meta.get(k)
                    for k in ("section", "label", "fetched_at", "source", "count")
                }
            )
        else:
            st.caption(f"Missing: {section_path}")
