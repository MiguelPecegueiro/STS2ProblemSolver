"""Map general card_id state to Sim 3 tag state for identical prune + memo keys."""

from __future__ import annotations

from combat_sim.sim3.tuple_dp import TupleStateSim3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from combat_sim.general.state import GeneralState

_TAG_BY_ID = {
    "STRIKE": "S",
    "DEFEND": "D",
    "BASH": "B",
    "BLOODLETTING": "L",
    "INFLAME": "I",
    "TWIN_STRIKE": "T",
}
from combat_sim.general.pile import pile_to_tags


def to_sim3_state(st: "GeneralState") -> TupleStateSim3:
    h = st.hand_dict()
    return TupleStateSim3(
        hp_p=st.hp_p,
        block_p=st.block_p,
        strength_p=st.strength_p,
        hp_e=st.hp_e,
        block_e=st.block_e,
        vuln_e=st.vuln_e,
        pattern_idx=st.pattern_idx,
        hand_s=h.get("STRIKE", 0),
        hand_d=h.get("DEFEND", 0),
        hand_b=h.get("BASH", 0),
        hand_bl=h.get("BLOODLETTING", 0),
        hand_inf=h.get("INFLAME", 0),
        draw=pile_to_tags(st.draw),
        discard=pile_to_tags(st.discard),
        turn=st.turn,
        shuffles=st.shuffles,
    )
