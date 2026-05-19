# General card engine (target architecture)

## Goal

**Any deck + any modeled enemy → optimal (or near-optimal) play.**

The Sim 0/1/2/3 stack validated math and pruning. The end state is one solver driven by **data**, not per-sim card-type counters.

```
Today:     Sim N → (hand_s, hand_d, hand_b, …) → bespoke DP
Target:    Deck → CardEffect[] → general engine → one DP
```

## Card model

Every card is a **state transition** described by `CardEffect` (see `combat_sim/card_effect.py`):

| Field | Role |
|-------|------|
| `cost`, `damage`, `hits` | Energy and combat |
| `block`, `hp_loss`, `energy_gain` | Defense / resource |
| `vuln_apply`, `weak_apply`, `frail_apply`, `strength_apply` | Status |
| `exhaust`, `draw` | Pile (draw = Phase 3) |
| `target` | `enemy` / `self` / `all_enemies` |

Cards are **data**, not branches in `tuple_dp.py`.

```python
STRIKE = CardEffect(card_id="STRIKE", cost=1, damage=6)
BASH = CardEffect(card_id="BASH", cost=2, damage=8, vuln_apply=2)
BLOODLETTING = CardEffect(card_id="BLOODLETTING", cost=0, hp_loss=3, energy_gain=2)
TWIN_STRIKE = CardEffect(card_id="TWIN_STRIKE", cost=1, damage=5, hits=2)
```

## Application layer

`combat_sim/card_engine.py` — `apply_card_effect(state, effect, target_enemy_id)`:

1. Pay cost / hp_loss / energy_gain  
2. For each hit: damage through `calc_damage` + block/shell  
3. Apply status stacks  
4. Block (with frail)  
5. Strength  
6. Discard or exhaust  

Same semantics as `CombatEngine.play_card`, but driven by `CardEffect`.

**Bridge:** `card_effect_from_card_def(CardDef)` and (later) `card_effect_from_codex(dict)` from `data/cards/`.

## Solver categories

### Category 1 — Deterministic (~60%)

Exact DP: `V*(s) = max_a V*(transition(s, a))`.

Current tuple DP + prunes apply. General DP will enumerate **legal multisets of instance plays per turn** (or aggregated counts when cards are indistinguishable).

### Category 2 — Random (~25%)

Expected value: `V*(s) = max_a E[V*(s')]`.

Examples: True Grit (random exhaust), Infernal Blade (random attack to hand).  
Enumerate small outcome sets; weighted max over children.

### Category 3 — Complex triggers (~15%)

`on_draw`, `on_exhaust`, `on_turn_start`, powers.  
Approximate or cap state; flag in card DB; MC verification.

## Phased delivery

### Phase 1 — General card engine (in progress)

- [x] `CardEffect` dataclass  
- [x] `apply_card_effect` on `CombatState`  
- [x] `card_effect_from_card_def` bridge  
- [x] Multi-hit (`hits > 1`)  
- [x] General tuple DP (`combat_sim/general/`) — card_id hand, play enumeration, `apply_turn`  
- [x] Regression vs Sim 3 on jaw worm, Effigy, Skulking (`tests/test_general_dp.py`)  
- [ ] Deck as `list[CardEffect]` + load from Codex tags  
- [ ] `solve_fight(deck, enemy, …)` from fight start (opening expectation)  
- [x] Twin Strike scenario + multi-hit Slow/vuln tests (`tests/test_twin_strike_general_dp.py`)  
- [ ] Uppercut scenario (weak + vuln in tuple DP)  

Sim 0–3 remain until broader regression suite passes, then deprecate.

### Play enumeration (Phase 1)

For hand multiset `H` and effect table `E`, enumerate every `play ⊆ H` with counts `0 ≤ play[c] ≤ H[c]`:

```
energy_gain(play) = Σ E[c].energy_gain × play[c]
energy_spent(play) = Σ E[c].cost × play[c]
legal iff energy_spent ≤ base_energy + energy_gain
```

Implementation: recursive over sorted card types (`combat_sim/general/plays.py::legal_plays`).

**Play order** (fixed rule, Sim 3–compatible): expand to a list, sort by  
`(-energy_gain, -block, -damage, card_id)` — energy first, block before attacks (Slow), Bash before Strike (vuln).

**Shuffle:** discard is sorted and shuffled via **tags** (`S`,`D`,…) so RNG matches Sim 3 (`pile.py::shuffle_discard_into_draw`).

### Phase 2 — Random (expected value)

- Random exhaust / generate  
- Weighted transitions in DP  

### Phase 3 — Triggers

- Trigger registry on fight state  
- Turn start / draw / exhaust hooks  

### Phase 4 — Intractable cards

- `approximate: true` in card metadata  
- Heuristic value or MC overlay  

## Migration from Sim 3

| Sim 3 concept | General equivalent |
|---------------|-------------------|
| `TurnCompositionSim3(strikes, defends, …)` | Multiset of plays from hand tags |
| `feasible_plays_sim3` | Energy-bounded subset sum on hand |
| `apply_turn_sim3` | Sequence of `apply_card_effect` + end turn |
| `TupleStateSim3` | HP, piles, statuses, pattern_idx, … (unchanged) |

**Indistinguishable cards:** keep aggregated counts (fast).  
**Distinct cards (upgraded, innate):** instance-based DP when needed.

## Data sources

| Source | Use |
|--------|-----|
| `data/cards/` | Card text, stats, Codex IDs → `CardEffect` (parser grows over time) |
| `data/monsters/` | Patterns, HP, moves → `EnemyState` / pattern tuple |
| `reference/monsters/` | Tier-A sim patterns until Codex parser is complete |

## Success criteria

1. Ironclad starter deck: general engine matches Sim 3 optimal on all scenario tests.  
2. Add Twin Strike / Uppercut via data only (no new `hand_*` field).  
3. Batch optimal fights in &lt;1s for typical Act 1 elites (with existing prunes).  
4. Dashboard/agent can point at same `CardEffect` definitions.

## Related docs

- `docs/IRONCLAD_CARD_ROADMAP.md` — mechanic steps (weak, draw, exhaust, …) inside Phase 1 engine  
- `docs/COMBAT_SIM.md` — current tuple DP, prunes, scenarios  
