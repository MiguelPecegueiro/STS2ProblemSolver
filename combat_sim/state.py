"""Combat state: player piles, enemies, intents."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Self

from combat_sim.cards import CardDef
from combat_sim.shuffle import shuffle_card_instances


class CombatPhase(str, Enum):
    PLAYER = "player"
    WON = "won"
    LOST = "lost"


class IntentKind(str, Enum):
    ATTACK = "attack"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class Intent:
    kind: IntentKind
    value: int
    enemy_block_bonus: int = 0

    @property
    def label(self) -> str:
        if self.kind == IntentKind.ATTACK:
            if self.enemy_block_bonus > 0:
                return f"Attack {self.value} (+{self.enemy_block_bonus} block)"
            return f"Attack {self.value}"
        return f"Block {self.value}"


@dataclass
class CardInstance:
    """One card in a pile (identity for play-order search)."""

    instance_id: int
    definition: CardDef

    @property
    def card_id(self) -> str:
        return self.definition.card_id

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def cost(self) -> int:
        return self.definition.cost


@dataclass
class EnemyState:
    enemy_id: str
    name: str
    hp: int
    max_hp: int
    block: int = 0
    vuln_stacks: int = 0
    weak_stacks: int = 0
    has_slow: bool = False
    slow_cards_this_turn: int = 0
    hp_loss_cap_per_turn: int | None = None
    hp_lost_this_player_turn: int = 0
    pattern: list[Intent] = field(default_factory=list)
    pattern_index: int = 0

    def copy(self) -> Self:
        return EnemyState(
            enemy_id=self.enemy_id,
            name=self.name,
            hp=self.hp,
            max_hp=self.max_hp,
            block=self.block,
            vuln_stacks=self.vuln_stacks,
            weak_stacks=self.weak_stacks,
            has_slow=self.has_slow,
            slow_cards_this_turn=self.slow_cards_this_turn,
            hp_loss_cap_per_turn=self.hp_loss_cap_per_turn,
            hp_lost_this_player_turn=self.hp_lost_this_player_turn,
            pattern=list(self.pattern),
            pattern_index=self.pattern_index,
        )

    def current_intent(self) -> Intent | None:
        if not self.pattern or self.hp <= 0:
            return None
        return self.pattern[self.pattern_index % len(self.pattern)]

    def is_alive(self) -> bool:
        return self.hp > 0

    def incoming_attack_damage(self) -> int:
        from combat_sim.damage_util import apply_weak_to_damage

        intent = self.current_intent()
        if intent and intent.kind == IntentKind.ATTACK:
            return apply_weak_to_damage(intent.value, self.weak_stacks)
        return 0


@dataclass
class CombatState:
    player_hp: int
    player_max_hp: int
    player_block: int = 0
    player_strength: int = 0
    frail_stacks: int = 0
    energy: int = 3
    hand: list[CardInstance] = field(default_factory=list)
    draw_pile: list[CardInstance] = field(default_factory=list)
    discard_pile: list[CardInstance] = field(default_factory=list)
    enemies: list[EnemyState] = field(default_factory=list)
    phase: CombatPhase = CombatPhase.PLAYER
    turn: int = 1
    _next_instance_id: int = 0
    shuffle_seed: int = 0
    shuffle_count: int = 0
    rng: random.Random | None = None  # legacy; reshuffles use canonical_shuffle

    def copy(self) -> Self:
        return CombatState(
            player_hp=self.player_hp,
            player_max_hp=self.player_max_hp,
            player_block=self.player_block,
            player_strength=self.player_strength,
            frail_stacks=self.frail_stacks,
            energy=self.energy,
            hand=[c for c in self.hand],
            draw_pile=[c for c in self.draw_pile],
            discard_pile=[c for c in self.discard_pile],
            enemies=[e.copy() for e in self.enemies],
            phase=self.phase,
            turn=self.turn,
            _next_instance_id=self._next_instance_id,
            shuffle_seed=self.shuffle_seed,
            shuffle_count=self.shuffle_count,
            rng=self.rng,
        )

    def living_enemies(self) -> list[EnemyState]:
        return [e for e in self.enemies if e.is_alive()]

    def total_incoming_damage(self) -> int:
        return sum(e.incoming_attack_damage() for e in self.living_enemies())

    def all_enemies_dead(self) -> bool:
        return not self.living_enemies()

    def allocate_instance_id(self) -> int:
        nid = self._next_instance_id
        self._next_instance_id += 1
        return nid

    @classmethod
    def new_fight(
        cls,
        *,
        deck: list[CardDef],
        player_hp: int = 80,
        enemies: list[EnemyState],
        seed: int | None = None,
    ) -> Self:
        shuffle_seed = 0 if seed is None else int(seed)
        instances: list[CardInstance] = []
        state = cls(
            player_hp=player_hp,
            player_max_hp=player_hp,
            enemies=[e.copy() for e in enemies],
            shuffle_seed=shuffle_seed,
            shuffle_count=0,
        )
        for card_def in deck:
            iid = state.allocate_instance_id()
            instances.append(CardInstance(instance_id=iid, definition=card_def))
        state.draw_pile = shuffle_card_instances(instances, shuffle_seed, 0)
        state.shuffle_count = 1
        state._begin_player_turn(first_turn=True)
        return state

    def _begin_player_turn(self, *, first_turn: bool = False) -> None:
        if not first_turn:
            self.player_block = 0
            for enemy in self.enemies:
                enemy.vuln_stacks = max(0, enemy.vuln_stacks - 1)
        self.energy = 3
        self._draw(5)
        if self.phase == CombatPhase.PLAYER and self.all_enemies_dead():
            self.phase = CombatPhase.WON

    def _draw(self, count: int) -> None:
        for _ in range(count):
            if not self.draw_pile:
                if not self.discard_pile:
                    break
                self.draw_pile = shuffle_card_instances(
                    self.discard_pile,
                    self.shuffle_seed,
                    self.shuffle_count,
                )
                self.discard_pile = []
                self.shuffle_count += 1
            if not self.draw_pile:
                break
            self.hand.append(self.draw_pile.pop())
