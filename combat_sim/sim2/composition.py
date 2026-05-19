"""Legal (strikes, defends, bash, bloodletting) plays for Sim 2."""

from __future__ import annotations

from dataclasses import dataclass

BASE_ENERGY = 3
ENERGY_PER_BLOODLETTING = 2


@dataclass(frozen=True, slots=True)
class TurnCompositionSim2:
    strikes: int
    defends: int
    bash: int = 0
    bloodletting: int = 0

    @property
    def energy_cost(self) -> int:
        """Energy spent on strikes/defends/bash (Bloodletting costs 0)."""
        return self.strikes + self.defends + 2 * self.bash

    @property
    def energy_available(self) -> int:
        return BASE_ENERGY + ENERGY_PER_BLOODLETTING * self.bloodletting

    @property
    def cards_played(self) -> int:
        return self.strikes + self.defends + self.bash + self.bloodletting

    def label(self) -> str:
        if self.cards_played == 0:
            return "End Turn (no plays)"
        parts: list[str] = []
        if self.bloodletting:
            parts.append(f"{self.bloodletting} Bloodletting")
        if self.strikes:
            parts.append(f"{self.strikes} Strike")
        if self.defends:
            parts.append(f"{self.defends} Defend")
        if self.bash:
            parts.append(f"{self.bash} Bash")
        return " + ".join(parts)


def feasible_plays_sim2(
    hand_s: int,
    hand_d: int,
    hand_b: int,
    hand_bl: int,
) -> list[TurnCompositionSim2]:
    """All (s,d,b,bl) with s+d+2b <= 3+2*bl and counts in hand."""
    out: list[TurnCompositionSim2] = []
    max_bl = min(hand_bl, 1)
    for bl in range(0, max_bl + 1):
        energy_cap = BASE_ENERGY + ENERGY_PER_BLOODLETTING * bl
        max_b = min(hand_b, 1)
        for b in range(0, max_b + 1):
            for s in range(0, hand_s + 1):
                for d in range(0, hand_d + 1):
                    if s + d + 2 * b <= energy_cap:
                        out.append(TurnCompositionSim2(s, d, b, bl))
    return out
