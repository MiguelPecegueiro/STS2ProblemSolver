"""Legal (strikes, defends, bash) plays for Sim 1."""

from __future__ import annotations

from dataclasses import dataclass

MAX_ENERGY = 3


@dataclass(frozen=True, slots=True)
class TurnCompositionSim1:
    strikes: int
    defends: int
    bash: int = 0

    @property
    def energy_cost(self) -> int:
        return self.strikes + self.defends + 2 * self.bash

    @property
    def cards_played(self) -> int:
        return self.strikes + self.defends + self.bash

    def label(self) -> str:
        if self.cards_played == 0:
            return "End Turn (no plays)"
        parts: list[str] = []
        if self.strikes:
            parts.append(f"{self.strikes} Strike")
        if self.defends:
            parts.append(f"{self.defends} Defend")
        if self.bash:
            parts.append(f"{self.bash} Bash")
        return " + ".join(parts)


def feasible_plays_sim1(hand_s: int, hand_d: int, hand_b: int) -> list[TurnCompositionSim1]:
    """All (s,d,b) with s+d+2b <= 3 and counts in hand."""
    out: list[TurnCompositionSim1] = []
    max_b = min(hand_b, 1)
    for b in range(0, max_b + 1):
        for s in range(0, hand_s + 1):
            for d in range(0, hand_d + 1):
                if s + d + 2 * b <= MAX_ENERGY:
                    out.append(TurnCompositionSim1(s, d, b))
    return out
