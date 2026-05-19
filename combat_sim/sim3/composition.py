"""Legal plays for Sim 3 (Sim 2 + Inflame)."""

from __future__ import annotations

from dataclasses import dataclass

BASE_ENERGY = 3
ENERGY_PER_BLOODLETTING = 2


@dataclass(frozen=True, slots=True)
class TurnCompositionSim3:
    strikes: int
    defends: int
    bash: int = 0
    bloodletting: int = 0
    inflame: int = 0

    @property
    def energy_cost(self) -> int:
        return self.strikes + self.defends + 2 * self.bash + self.inflame

    @property
    def energy_available(self) -> int:
        return BASE_ENERGY + ENERGY_PER_BLOODLETTING * self.bloodletting

    @property
    def cards_played(self) -> int:
        return self.strikes + self.defends + self.bash + self.bloodletting + self.inflame

    def label(self) -> str:
        if self.cards_played == 0:
            return "End Turn (no plays)"
        parts: list[str] = []
        if self.bloodletting:
            parts.append(f"{self.bloodletting} Bloodletting")
        if self.inflame:
            parts.append(f"{self.inflame} Inflame")
        if self.strikes:
            parts.append(f"{self.strikes} Strike")
        if self.defends:
            parts.append(f"{self.defends} Defend")
        if self.bash:
            parts.append(f"{self.bash} Bash")
        return " + ".join(parts)


def feasible_plays_sim3(
    hand_s: int,
    hand_d: int,
    hand_b: int,
    hand_bl: int,
    hand_inf: int,
) -> list[TurnCompositionSim3]:
    """All (s,d,b,bl,inf) with s+d+2b+inf <= 3+2*bl and counts in hand."""
    out: list[TurnCompositionSim3] = []
    max_bl = min(hand_bl, 1)
    for bl in range(0, max_bl + 1):
        energy_cap = BASE_ENERGY + ENERGY_PER_BLOODLETTING * bl
        max_inf = min(hand_inf, 1)
        for inf in range(0, max_inf + 1):
            max_b = min(hand_b, 1)
            for b in range(0, max_b + 1):
                for s in range(0, hand_s + 1):
                    for d in range(0, hand_d + 1):
                        if s + d + 2 * b + inf <= energy_cap:
                            out.append(TurnCompositionSim3(s, d, b, bl, inf))
    return out
