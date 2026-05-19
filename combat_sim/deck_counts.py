"""Count Strike/Defend cards in piles (for composition-based math solver)."""

from __future__ import annotations

from dataclasses import dataclass

from combat_sim.state import CardInstance, CombatState


@dataclass(frozen=True, slots=True)
class DeckCounts:
    strikes: int
    defends: int

    @property
    def total(self) -> int:
        return self.strikes + self.defends


def count_pile(cards: list[CardInstance]) -> DeckCounts:
    strikes = sum(1 for c in cards if c.definition.damage > 0)
    defends = sum(1 for c in cards if c.definition.block > 0)
    return DeckCounts(strikes, defends)


def hand_counts(state: CombatState) -> DeckCounts:
    return count_pile(state.hand)


def deck_totals(state: CombatState) -> DeckCounts:
    h = count_pile(state.hand)
    p = count_pile(state.draw_pile)
    q = count_pile(state.discard_pile)
    return DeckCounts(h.strikes + p.strikes + q.strikes, h.defends + p.defends + q.defends)


def format_hand_counts(state: CombatState) -> str:
    c = hand_counts(state)
    return f"{c.strikes} Strike, {c.defends} Defend"


@dataclass(frozen=True, slots=True)
class DeckCountsSim1:
    strikes: int
    defends: int
    bash: int

    @property
    def total(self) -> int:
        return self.strikes + self.defends + self.bash


def _is_strike(card: CardInstance) -> bool:
    return card.definition.card_id == "STRIKE"


def _is_bash(card: CardInstance) -> bool:
    return card.definition.card_id == "BASH"


def _is_defend(card: CardInstance) -> bool:
    return card.definition.block > 0 and not _is_bash(card)


def count_pile_sim1(cards: list[CardInstance]) -> DeckCountsSim1:
    return DeckCountsSim1(
        sum(1 for c in cards if _is_strike(c)),
        sum(1 for c in cards if _is_defend(c)),
        sum(1 for c in cards if _is_bash(c)),
    )


def hand_counts_sim1(state: CombatState) -> DeckCountsSim1:
    return count_pile_sim1(state.hand)


def deck_totals_sim1(state: CombatState) -> DeckCountsSim1:
    h = count_pile_sim1(state.hand)
    p = count_pile_sim1(state.draw_pile)
    q = count_pile_sim1(state.discard_pile)
    return DeckCountsSim1(
        h.strikes + p.strikes + q.strikes,
        h.defends + p.defends + q.defends,
        h.bash + p.bash + q.bash,
    )


def format_hand_counts_sim1(state: CombatState) -> str:
    c = hand_counts_sim1(state)
    parts = [f"{c.strikes} Strike", f"{c.defends} Defend"]
    if c.bash:
        parts.append(f"{c.bash} Bash")
    return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class DeckCountsSim2:
    strikes: int
    defends: int
    bash: int
    bloodletting: int

    @property
    def total(self) -> int:
        return self.strikes + self.defends + self.bash + self.bloodletting


def _is_bloodletting(card: CardInstance) -> bool:
    return card.definition.card_id == "BLOODLETTING"


def count_pile_sim2(cards: list[CardInstance]) -> DeckCountsSim2:
    return DeckCountsSim2(
        sum(1 for c in cards if _is_strike(c)),
        sum(1 for c in cards if _is_defend(c)),
        sum(1 for c in cards if _is_bash(c)),
        sum(1 for c in cards if _is_bloodletting(c)),
    )


def hand_counts_sim2(state: CombatState) -> DeckCountsSim2:
    return count_pile_sim2(state.hand)


def deck_totals_sim2(state: CombatState) -> DeckCountsSim2:
    h = count_pile_sim2(state.hand)
    p = count_pile_sim2(state.draw_pile)
    q = count_pile_sim2(state.discard_pile)
    return DeckCountsSim2(
        h.strikes + p.strikes + q.strikes,
        h.defends + p.defends + q.defends,
        h.bash + p.bash + q.bash,
        h.bloodletting + p.bloodletting + q.bloodletting,
    )


def format_hand_counts_sim2(state: CombatState) -> str:
    c = hand_counts_sim2(state)
    parts = [f"{c.strikes} Strike", f"{c.defends} Defend"]
    if c.bash:
        parts.append(f"{c.bash} Bash")
    if c.bloodletting:
        parts.append(f"{c.bloodletting} Bloodletting")
    return ", ".join(parts)


@dataclass(frozen=True, slots=True)
class DeckCountsSim3:
    strikes: int
    defends: int
    bash: int
    bloodletting: int
    inflame: int

    @property
    def total(self) -> int:
        return self.strikes + self.defends + self.bash + self.bloodletting + self.inflame


def _is_inflame(card: CardInstance) -> bool:
    return card.definition.card_id == "INFLAME"


def count_pile_sim3(cards: list[CardInstance]) -> DeckCountsSim3:
    return DeckCountsSim3(
        sum(1 for c in cards if _is_strike(c)),
        sum(1 for c in cards if _is_defend(c)),
        sum(1 for c in cards if _is_bash(c)),
        sum(1 for c in cards if _is_bloodletting(c)),
        sum(1 for c in cards if _is_inflame(c)),
    )


def hand_counts_sim3(state: CombatState) -> DeckCountsSim3:
    return count_pile_sim3(state.hand)


def deck_totals_sim3(state: CombatState) -> DeckCountsSim3:
    h = count_pile_sim3(state.hand)
    p = count_pile_sim3(state.draw_pile)
    q = count_pile_sim3(state.discard_pile)
    return DeckCountsSim3(
        h.strikes + p.strikes + q.strikes,
        h.defends + p.defends + q.defends,
        h.bash + p.bash + q.bash,
        h.bloodletting + p.bloodletting + q.bloodletting,
        h.inflame + p.inflame + q.inflame,
    )


def format_hand_counts_sim3(state: CombatState) -> str:
    c = hand_counts_sim3(state)
    parts = [f"{c.strikes} Strike", f"{c.defends} Defend"]
    if c.bash:
        parts.append(f"{c.bash} Bash")
    if c.bloodletting:
        parts.append(f"{c.bloodletting} Bloodletting")
    if c.inflame:
        parts.append(f"{c.inflame} Inflame")
    if state.player_strength:
        parts.append(f"Str {state.player_strength}")
    return ", ".join(parts)
