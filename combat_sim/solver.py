"""Greedy turn solver via exhaustive play-sequence search."""

from __future__ import annotations

from dataclasses import dataclass

from combat_sim.engine import CombatEngine, TurnAction
from combat_sim.state import CombatPhase, CombatState


@dataclass(frozen=True, slots=True)
class TurnScore:
    lethal: bool
    hp_lost: int
    damage_dealt: int
    block_gained: int
    energy_left: int


def solve_turn(state: CombatState) -> TurnAction:
    """Pick a strong line for the current player turn."""
    if state.phase != CombatPhase.PLAYER:
        return TurnAction()

    living = state.living_enemies()
    default_target = living[0].enemy_id if len(living) == 1 else None

    best_action = TurnAction()
    best_key: tuple | None = None

    for plays in _enumerate_play_sequences(state, default_target):
        trial = state.copy()
        hp_before = trial.player_hp
        damage_before = sum(e.hp for e in trial.enemies)

        for instance_id, target_id in plays:
            CombatEngine.play_card(trial, instance_id, target_id)
            if trial.phase != CombatPhase.PLAYER:
                break

        if trial.phase == CombatPhase.WON:
            score = TurnScore(
                lethal=True,
                hp_lost=0,
                damage_dealt=999,
                block_gained=0,
                energy_left=trial.energy,
            )
        elif trial.phase == CombatPhase.PLAYER:
            CombatEngine._enemy_turn(trial)
            hp_lost = max(0, hp_before - trial.player_hp)
            damage_dealt = damage_before - sum(e.hp for e in trial.enemies)
            score = TurnScore(
                lethal=trial.all_enemies_dead(),
                hp_lost=hp_lost,
                damage_dealt=damage_dealt,
                block_gained=0,
                energy_left=trial.energy,
            )
        else:
            score = TurnScore(
                lethal=False,
                hp_lost=hp_before,
                damage_dealt=damage_before - sum(e.hp for e in trial.enemies),
                block_gained=0,
                energy_left=0,
            )

        key = _score_key(score)
        if best_key is None or key > best_key:
            best_key = key
            best_action = TurnAction(plays=plays)

    return best_action


def solve_fight(state: CombatState, *, max_turns: int = 30):
    """Run a full fight using :func:`solve_turn` each player turn."""
    from combat_sim.engine import FightResult

    return CombatEngine.run_fight(state, solve_turn, max_turns=max_turns)


def _score_key(score: TurnScore) -> tuple:
    return (
        1 if score.lethal else 0,
        -score.hp_lost,
        score.damage_dealt,
        score.block_gained,
        score.energy_left,
    )


def _enumerate_play_sequences(
    state: CombatState,
    default_target: str | None,
) -> list[tuple[tuple[int, str | None], ...]]:
    sequences: list[tuple[tuple[int, str | None], ...]] = []

    def dfs(
        sim: CombatState,
        plays: list[tuple[int, str | None]],
        used: frozenset[int],
    ) -> None:
        sequences.append(tuple(plays))
        living = sim.living_enemies()
        if not living:
            return
        target = default_target or (living[0].enemy_id if len(living) == 1 else None)

        for idx, card in enumerate(sim.hand):
            if card.instance_id in used:
                continue
            if card.cost > sim.energy:
                continue
            options: list[tuple[int, str | None]] = [(card.instance_id, None)]
            if card.definition.damage > 0:
                if target:
                    options = [(card.instance_id, target)]
                else:
                    options = [
                        (card.instance_id, enemy.enemy_id) for enemy in living
                    ]
            elif card.definition.block <= 0:
                continue

            for play in options:
                nxt = sim.copy()
                if not CombatEngine.play_card(nxt, play[0], play[1]):
                    continue
                if nxt.phase != CombatPhase.PLAYER:
                    sequences.append(tuple(plays + [play]))
                    continue
                dfs(nxt, plays + [play], used | {card.instance_id})

    dfs(state, [], frozenset())
    return sequences
