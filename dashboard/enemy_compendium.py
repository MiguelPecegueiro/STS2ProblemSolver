"""Dashboard for agent-learned enemy compendium (data/enemy_compendium.json)."""

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

from sts2_agent.enemy_compendium import (  # noqa: E402
    COMPENDIUM_PATH,
    group_compendium_by_encounter,
    reload_compendium,
)

FILTER_COMPENDIUM_ENCOUNTER = "dash_filter_compendium_encounter"
FILTER_COMPENDIUM_SLOT = "dash_filter_compendium_slot"


def _ensure_select_value(key: str, options: list, default) -> None:
    if key not in st.session_state:
        st.session_state[key] = default
    elif options and st.session_state[key] not in options:
        st.session_state[key] = default if default in options else options[0]


def _load() -> dict:
    if not COMPENDIUM_PATH.exists():
        return {"version": 1, "enemies": {}}
    return json.loads(COMPENDIUM_PATH.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    COMPENDIUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPENDIUM_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    reload_compendium()


def _render_entry(entry: dict, storage_key: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Fights recorded", int(entry.get("fight_count") or 0))
    cycle = entry.get("learned_cycle") or []
    c2.metric("Learned cycle length", len(cycle))
    c3.metric("Distinct moves", len(entry.get("moves") or {}))

    if entry.get("role") or entry.get("entity_id_sample"):
        st.caption(
            f"Storage key: `{storage_key}` · role: `{entry.get('role') or '-'}` · "
            f"sample entity: `{entry.get('entity_id_sample') or '-'}`"
        )

    st.subheader("Learned turn order (this slot)")
    if cycle:
        st.code(" → ".join(cycle))
    else:
        st.caption("Need at least one full combat on this slot to infer a cycle.")

    st.subheader("Moves")
    rows = []
    for mk, mv in (entry.get("moves") or {}).items():
        if not isinstance(mv, dict):
            continue
        verified = mv.get("verified_runs") or []
        rows.append(
            {
                "move_key": mk,
                "api_label": mv.get("api_label"),
                "damage": mv.get("damage"),
                "block": mv.get("block"),
                "tags": ", ".join(mv.get("tags") or []),
                "seen": mv.get("seen_count"),
                "verified_runs": len(verified),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No moves recorded yet.")

    st.subheader("Recent sequences (this slot)")
    for seq in reversed((entry.get("sequences") or [])[-5:]):
        if isinstance(seq, dict):
            eid = str(seq.get("entity_id") or "")[:24]
            st.text(
                f"run {str(seq.get('run_id', ''))[:8]}… "
                f"{f'entity {eid}…' if eid else ''}: {' → '.join(seq.get('moves') or [])}"
            )


def render_compendium() -> None:
    st.header("Enemy compendium")
    st.caption(
        f"Learned per slot (front/middle/back, twins, etc.). `{COMPENDIUM_PATH.relative_to(PROJECT_ROOT)}`"
    )

    if st.button("Reload", key="reload_learned"):
        st.cache_data.clear()
        st.rerun()

    data = _load()
    enemies = data.get("enemies") or {}

    if not enemies:
        st.info("Empty - run the agent through fights to fill this file.")
        return

    groups = group_compendium_by_encounter(enemies)
    st.metric("Encounter types", len(groups))
    st.metric("Compendium entries (slots)", len(enemies))

    encounter_names = sorted(
        groups.keys(),
        key=lambda k: str((enemies.get(groups[k][0]) or {}).get("name") or k),
    )
    _ensure_select_value(
        FILTER_COMPENDIUM_ENCOUNTER,
        encounter_names,
        encounter_names[0],
    )
    encounter_pick = st.selectbox(
        "Encounter type",
        encounter_names,
        format_func=lambda k: (enemies.get(groups[k][0]) or {}).get("base_name") or k,
        key=FILTER_COMPENDIUM_ENCOUNTER,
    )
    slot_keys = groups.get(encounter_pick) or []

    if len(slot_keys) > 1:
        st.subheader("Slots in this encounter")
        overview = []
        for sk in slot_keys:
            ent = enemies.get(sk) or {}
            overview.append(
                {
                    "slot": sk,
                    "display": ent.get("name"),
                    "role": ent.get("role") or "-",
                    "fights": ent.get("fight_count"),
                    "moves": len(ent.get("moves") or {}),
                    "cycle_len": len(ent.get("learned_cycle") or []),
                }
            )
        st.dataframe(pd.DataFrame(overview), use_container_width=True, hide_index=True)

    if slot_keys:
        _ensure_select_value(FILTER_COMPENDIUM_SLOT, slot_keys, slot_keys[0])
    pick = st.selectbox(
        "Compendium entry (slot)",
        slot_keys,
        format_func=lambda k: (enemies.get(k) or {}).get("name") or k,
        key=FILTER_COMPENDIUM_SLOT,
    )
    entry = enemies[pick]

    _render_entry(entry, pick)

    with st.expander("Raw JSON (advanced)"):
        st.json(entry)

    if st.button("Delete this compendium entry", type="secondary"):
        del enemies[pick]
        data["enemies"] = enemies
        _save(data)
        st.success("Deleted")
        st.rerun()
