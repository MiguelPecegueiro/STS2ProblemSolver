"""Dashboard — browse Spire Codex monsters from data/monsters/ by type section."""

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

from sts2_agent.codex_monsters import (  # noqa: E402
    MONSTER_SECTIONS,
    MONSTERS_ROOT,
    SECTION_LABELS,
    load_all_monsters,
    load_monster_index,
)

FILTER_MONSTER_TYPES = "dash_filter_monster_types"
FILTER_MONSTER_ACTS = "dash_filter_monster_acts"
FILTER_MONSTER_SEARCH = "dash_filter_monster_search"
FILTER_MONSTER_PICK = "dash_filter_monster_pick"


@st.cache_data(ttl=300)
def _cached_index() -> dict:
    return load_monster_index()


@st.cache_data(ttl=300)
def _cached_all_monsters() -> list[dict]:
    return load_all_monsters()


def _monster_summary_row(monster: dict) -> dict:
    pattern = monster.get("attack_pattern") or {}
    return {
        "name": monster.get("name"),
        "id": monster.get("id"),
        "section": monster.get("section") or monster.get("type"),
        "min_hp": monster.get("min_hp"),
        "max_hp": monster.get("max_hp"),
        "moves": len(monster.get("moves") or []),
        "act": monster.get("primary_act"),
        "pattern": pattern.get("description"),
    }


def _move_rows(monster: dict) -> list[dict]:
    rows: list[dict] = []
    for move in monster.get("moves") or []:
        if not isinstance(move, dict):
            continue
        dmg = move.get("damage")
        dmg_n = None
        hits = None
        if isinstance(dmg, dict):
            dmg_n = dmg.get("normal")
            hits = dmg.get("hit_count")
        rows.append(
            {
                "id": move.get("id"),
                "name": move.get("name"),
                "intent": move.get("intent"),
                "damage": dmg_n,
                "hits": hits,
                "block": move.get("block"),
                "heal": move.get("heal"),
            }
        )
    return rows


def _ensure_pick(key: str, options: list[str], default: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = default
    elif options and st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]


def render_monster_compendium() -> None:
    st.header("Enemy compendium")
    st.caption(
        f"Spire Codex monsters by type (Normal / Elite / Boss). "
        f"Source: `{MONSTERS_ROOT.relative_to(PROJECT_ROOT)}`"
    )

    if st.button("Reload monsters", key="reload_monster_compendium"):
        st.cache_data.clear()
        st.rerun()

    index = _cached_index()
    if not index:
        st.warning("No local monster database found. Import from Spire Codex first:")
        st.code("py tools/import_codex_monsters.py --force", language="powershell")
        return

    all_monsters = _cached_all_monsters()
    if not all_monsters:
        st.info("Index exists but no monster files loaded.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total monsters", int(index.get("total_monsters") or len(all_monsters)))
    c2.metric("Sections", len(index.get("sections") or {}))
    c3.metric("Fetched (UTC)", str(index.get("fetched_at") or "")[:19])
    c4.metric("Unmapped", int(index.get("unmapped_count") or 0))

    section_keys = [k for k, _ in MONSTER_SECTIONS]
    section_labels = {k: SECTION_LABELS[k] for k in section_keys}
    counts = {
        k: int((index.get("sections") or {}).get(k, {}).get("count") or 0)
        for k in section_keys
    }

    with st.expander("Monsters per section", expanded=False):
        overview = pd.DataFrame(
            [
                {"section": section_labels[k], "key": k, "count": counts.get(k, 0)}
                for k in section_keys
            ]
        )
        st.dataframe(overview, use_container_width=True, hide_index=True)

    act_options = sorted(
        {str(m.get("primary_act")) for m in all_monsters if m.get("primary_act")}
    )

    if FILTER_MONSTER_TYPES not in st.session_state:
        st.session_state[FILTER_MONSTER_TYPES] = section_keys

    type_filter = st.multiselect(
        "Type filter",
        options=section_keys,
        format_func=lambda k: f"{section_labels[k]} ({counts.get(k, 0)})",
        key=FILTER_MONSTER_TYPES,
    )
    if not type_filter:
        st.info("Select at least one type section to browse.")
        return

    if FILTER_MONSTER_ACTS not in st.session_state:
        st.session_state[FILTER_MONSTER_ACTS] = act_options

    act_filter = st.multiselect(
        "Act filter (optional)",
        options=act_options,
        key=FILTER_MONSTER_ACTS,
    )

    search = st.text_input(
        "Search (name, id, move, pattern)",
        placeholder="e.g. Skulking, SMASH, Underdocks",
        key=FILTER_MONSTER_SEARCH,
    ).strip().lower()

    filtered: list[dict] = []
    for monster in all_monsters:
        section = str(monster.get("section") or monster.get("type") or "").lower()
        if section not in type_filter:
            continue
        act = monster.get("primary_act")
        if act_filter and act not in act_filter:
            continue
        if search:
            blob = " ".join(
                str(monster.get(k) or "")
                for k in ("name", "id", "primary_act")
            ).lower()
            pattern = monster.get("attack_pattern") or {}
            blob += " " + str(pattern.get("description") or "").lower()
            for move in monster.get("moves") or []:
                if isinstance(move, dict):
                    blob += " " + str(move.get("name") or "").lower()
                    blob += " " + str(move.get("id") or "").lower()
            if search not in blob:
                continue
        filtered.append(monster)

    st.caption(f"Showing **{len(filtered)}** of {len(all_monsters)} monsters")

    if not filtered:
        st.info("No monsters match filters.")
        return

    table = pd.DataFrame(_monster_summary_row(m) for m in filtered)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "min_hp": st.column_config.NumberColumn(format="%d"),
            "max_hp": st.column_config.NumberColumn(format="%d"),
            "moves": st.column_config.NumberColumn(format="%d"),
        },
    )

    pick_labels = [
        f"{m.get('name')} [{m.get('id')}] ({section_labels.get(str(m.get('section') or ''), m.get('section'))})"
        for m in filtered
    ]
    pick_ids = [str(m.get("id") or m.get("name")) for m in filtered]

    default_id = pick_ids[0]
    if FILTER_MONSTER_PICK in st.session_state:
        prev = st.session_state[FILTER_MONSTER_PICK]
        if prev in pick_ids:
            default_id = prev

    _ensure_pick(FILTER_MONSTER_PICK, pick_ids, default_id)
    selected_id = st.selectbox(
        "Open monster (exact Codex JSON)",
        pick_ids,
        format_func=lambda mid: pick_labels[pick_ids.index(mid)],
        key=FILTER_MONSTER_PICK,
    )

    selected = next(
        m for m in filtered if str(m.get("id") or m.get("name")) == selected_id
    )
    section = str(selected.get("section") or selected.get("type") or "")

    st.subheader(str(selected.get("name") or selected_id))
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ID", str(selected.get("id") or "—"))
    m2.metric("Type", str(selected.get("type") or "—"))
    m3.metric("HP", f"{selected.get('min_hp') or '?'}-{selected.get('max_hp') or '?'}")
    m4.metric("Moves", len(selected.get("moves") or []))
    m5.metric("Section", section_labels.get(section, section))

    if selected.get("primary_act"):
        st.caption(f"Primary act: **{selected.get('primary_act')}**")

    pattern = selected.get("attack_pattern") or {}
    if pattern.get("description"):
        st.markdown(f"**Pattern:** {pattern['description']}")

    move_rows = _move_rows(selected)
    if move_rows:
        st.subheader("Moves")
        st.dataframe(pd.DataFrame(move_rows), use_container_width=True, hide_index=True)

    if selected.get("encounters"):
        with st.expander("Encounters"):
            st.dataframe(
                pd.DataFrame(selected["encounters"]),
                use_container_width=True,
                hide_index=True,
            )

    if selected.get("innate_powers"):
        with st.expander("Innate powers"):
            st.json(selected["innate_powers"])

    st.subheader("Exact JSON entry")
    st.caption(
        "Full object stored under `data/monsters/by_type/<section>.json` — use for sim / agent work."
    )
    st.json(selected)

    with st.expander("Copy as formatted JSON"):
        st.code(json.dumps(selected, ensure_ascii=False, indent=2), language="json")

    with st.expander("Section file metadata"):
        section_path = MONSTERS_ROOT / "by_type" / f"{section}.json"
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
