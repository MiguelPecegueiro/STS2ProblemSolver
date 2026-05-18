# Dashboard specification

Streamlit app: `dashboard/app.py`. Analytics logic: `dashboard/metrics.py`, UI blocks: `dashboard/sections.py`.

Run from repo root:

```bash
streamlit run dashboard/app.py
```

---

## Tabs

| Tab | Purpose |
|-----|---------|
| **Run analytics** | Phase A metrics (four sections below) |
| **Enemy compendium** | Unchanged — browse/edit learned enemy data |

---

## Sidebar filters

| Filter | Effect |
|--------|--------|
| **View** | Human / Agent / Compare / Compare versions |
| **Versions for overview (section 1)** | Multiselect — health trends / early warnings for all checked versions |
| **Agent version to analyze** | Below section 1 — single version for sections 2–4 (defaults to newest) |
| **Character** | Filter runs |
| **Ascension** | Filter runs |
| **Date** | Presets or custom range |

All runs are mixed in trend/comparison charts (no per-character normalization in Phase A).

---

## Section 1 — Health at a glance

| Metric | Source | Notes |
|--------|--------|-------|
| Summary table | `runs.jsonl` | Per `agent_version`: runs, avg floor, avg score, avg duration |
| **game_version** | `runs.jsonl` | Shown in section caption when present |
| Trend lines | `runs.jsonl` | Rolling avg **floor** and **score**, last 50 runs per version (two charts) |
| Early warning | `runs.jsonl` | Any version ≥**15%** lower avg floor than another at same run count (min 5 runs, window 50) |

**Removed (Phase A):** rolling win rate over time, run timing section, recent runs table.

---

## Section 2 — Where and why it dies

| Metric | Source | Notes |
|--------|--------|-------|
| Floor histogram | `runs.jsonl` | Losses only |
| Death category | `cause_of_death` | Elite / boss / monster / event / … |
| Enemy killer | `cause_of_death` or `killing_enemy` | Phase A: parse string; **Phase B data:** `killing_enemy.name` on new runs |
| Act breakdown | `act_reached` | Bar chart |
| HP entering act | `decisions.jsonl` | First `player_hp` snapshot per act |
| Turns per fight | `combat_summary[].turns` or decisions | **New runs:** structured; old runs: decision-count approximation |
| Damage per fight | `combat_summary[]` or hp lists | **New runs:** `damage_taken` / `damage_dealt` per fight |
| Human benchmark | `runs.jsonl` | Compare agent **run-total** `total_damage_taken` to human mean |

Old runs without `combat_summary` / `killing_enemy` should show **no data** in UI (follow-up), not errors.

---

## Section 3 — Card decisions

| Metric | Source | Notes |
|--------|--------|-------|
| Pick rates | Agent decisions + human `card_choices.jsonl` | Aggregate counts |
| Pick tiers | Mobalytics | Agent picked cards |
| Tier miss (human) | `card_choices` offered vs picked | ≥1 letter worse than best offered |
| Tier miss (agent) | `card_reward_offered` on decisions | **New runs:** compare pick to offered tiers |
| Matched offers | `card_reward_offered` | **New runs:** agent vs human on same offer set (UI follow-up) |
| Deck correlation | `final_deck` | Win vs loss |

---

## Section 4 — Combat efficiency

| Metric | Definition | Source |
|--------|------------|--------|
| Block efficiency | % combat turns with block when incoming intent > 0 | `decisions.jsonl` |
| Potion hoarding at death | % deaths with empty potion slot | `runs.jsonl` |
| Energy on end_turn | % `end_turn` with energy left | `decisions.jsonl` |
| Damage per fight | Per-fight damage dealt | `combat_summary[].damage_dealt` on new runs |

---

## Phase B logging (implemented)

Logged by `sts2_agent/data_pipeline.py` on **new agent runs** after deploy.

### 1. `card_reward_offered` (decisions)

On every `card_reward` decision (pick or skip):

- Top-level field: `card_reward_offered: string[]` (card names, 3 entries when screen shows 3)
- Duplicated on `state_snapshot.card_reward_offered` for training snapshots

### 2. `combat_summary[]` (runs)

Appended at end of each fight on `runs.jsonl`:

| Field | Type | Meaning |
|-------|------|---------|
| `enemy_names` | `string[]` | Names at fight start |
| `turns` | `int` | Agent combat decisions in fight |
| `damage_taken` | `int` | `hp_start - hp_end` (clamped ≥ 0) |
| `damage_dealt` | `int` | Sum of enemy HP lost this fight |
| `hp_start` | `int` | Player HP at combat start |
| `hp_end` | `int` | Player HP when combat ends |
| `won_fight` | `bool` | Player survived |
| `state_type` | `string` | `monster` / `elite` / `boss` / `hand_select` |

### 3. `killing_enemy` (runs)

On run end when the player lost in combat:

```json
{
  "name": "Jaw Worm",
  "entity_id": "JW_0",
  "compendium_key": "jaw_worm",
  "intent": "Attack"
}
```

`cause_of_death` remains for backwards compatibility (string includes killer name when known).

---

## UI follow-up (not yet done)

Dashboard sections still use Phase A derivations where structured fields are missing. Next pass:

- Enemy killer chart from `killing_enemy.name`
- Turns/damage charts from `combat_summary[]`
- Agent tier-miss and matched-offer tables from `card_reward_offered`
- Graceful “no data” captions for pre–Phase B runs

See also [PATCH_MANAGEMENT.md](PATCH_MANAGEMENT.md) for `game_version` tagging.
