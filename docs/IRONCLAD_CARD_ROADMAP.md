# Ironclad card roadmap (combat_sim)

Mechanic steps for the **general card engine** (`docs/GENERAL_CARD_ENGINE.md`).  
Sim 0–3 tuple DP is legacy; new cards should land as `CardEffect` data first.

Implementation order for expanding beyond the current Sim 0–3 deck. Each step should land with tests before the next.

## Status

| Step | Mechanic | Unlocks (examples) | Status |
|------|----------|-------------------|--------|
| — | Strike, Defend, Bash, BL, Inflame, Vuln, Slow, Shell | Sim 0–3 | Done |
| 1 | Weak + Frail | Uppercut, enemy attack reduction | Done (engine) |
| 2 | Multi-hit | Twin Strike, Dismantle, Sword Boomerang | Pending |
| 3 | Card draw | Pommel Strike, Shrug It Off, Offering | Pending |
| 4 | Exhaust pile | True Grit, Fiend Fire, Ashen Strike | Pending |
| 5 | Turn-local flags | Rage, Evil Eye, Conflagration | Pending |
| 6 | X-cost | Whirlwind, Cascade | Pending |
| 7 | Temp strength reset | Setup Strike, Demon Form | Pending |

## Step 1 — Weak + Frail

State:

- `enemy.weak_stacks` — outgoing attack damage × 0.75 (floor)
- `player.frail_stacks` — block gained × 0.75 (floor)

Decay: −1 at end of owner's turn (frail after player turn, weak after enemy turn).

Cards: `weak_apply` / `frail_apply` on `CardDef`; engine applies on play.

## Step 2 — Multi-hit

`damage_total = sum over hits of calc_damage(...)`.

## Step 3 — Draw

`hand' = hand ∪ draw[:n]`, `draw' = draw[n:]`.

## Step 4 — Exhaust pile

Track exhausted cards; `exhaust_count` for payoffs.

## Step 5 — Turn-local

`attacks_played_this_turn`, `exhausted_this_turn`, `hp_lost_this_turn`.

## Step 6 — X-cost

`cost = energy_remaining`, variable hits/damage.

## Step 7 — Temp strength

`strength_temp` reset end of turn.

## Out of scope (for now)

Multiplayer (Tank), pile manipulation at scale (Havoc, Rampage), random generation (Infernal Blade), full AoE multi-enemy.

See card tier list in project notes; ~40 cards reachable after steps 1–7 without pile manipulation.
