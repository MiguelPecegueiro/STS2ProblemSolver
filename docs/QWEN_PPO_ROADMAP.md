# Qwen + PPO roadmap

Strategic plan for a **self-improving** agent — not “bolt Qwen onto PPO for a few more floors,” but a living system that **evolves its own knowledge** each cycle. Near term: add a **Qwen** reasoning layer on top of **BC / PPO**. Long term: close the loop so discoveries in play rewrite `expert_knowledge.json`, retrain the foundation, and explore again.

**Related:** [Model checkpoint lineage](MODEL_LINEAGE.md) (current `ppo_v3`, BC retrain history).

---

## Long-term vision — self-improving system

The goal is **not** a static model shipped once. It is a **living, evolving system**:

- Each cycle **generates new knowledge** that feeds the next cycle.
- The system can surface **novel strategies** no human guide has documented.
- Human-curated tiers (Mobalytics, guides) are the **bootstrap**, not the ceiling.

Qwen integration is **phase one** of that loop — giving PPO a strategic channel. The full loop is what removes the theoretical cap.

### The knowledge generation loop

```mermaid
flowchart LR
  PPO["PPO explores<br/>action space"]
  Discover["Novel successful<br/>strategies emerge"]
  Doc["Analyze + document<br/>expert_knowledge.json"]
  Qwen["Qwen guides PPO<br/>toward new lines"]
  PPO

  PPO --> Discover --> Doc --> Qwen --> PPO
```

| Step | What happens |
|------|----------------|
| 1 | **PPO** explores the probability space with minimal human bias in the action head |
| 2 | **Novel successes** appear — runs that work but contradict or extend what Qwen / the knowledge base currently believe |
| 3 | Those outcomes are **analyzed and written** into `cache/expert_knowledge.json` (synergies, contextual tiers, archetype notes) |
| 4 | **Qwen** reads the updated base and **intentionally steers** the next wave of exploration |
| 5 | **PPO** probes variations of those strategies; new discoveries emerge |
| 6 | **Repeat** — each plateau becomes a signal to **expand the knowledge base**, not to stop |

This is the core product: **documentation and strategy discovery as a first-class output**, not only higher floor numbers on a fixed policy.

### Why this has no theoretical ceiling (in principle)

| Limitation | How the loop addresses it |
|------------|---------------------------|
| Pure PPO hits **local optima** | It only optimizes what the **195 (+N) features** can represent |
| Fixed human knowledge caps advice | **Living** `expert_knowledge.json` grows after each discovery cycle |
| Plateau on a given feature set | Plateau → **expand knowledge** → new features / priors → **new regions** for PPO to explore |
| Distillation of human play only | System **surpasses and extends** human docs — it does not stop at them |

In practice, balance patches, compute, and patch cadence still bound real-world progress. Philosophically, every plateau is **“expand the map,”** not **“we’re done.”**

### Five components — how they feed each other

```mermaid
flowchart TB
  Human["Human runs<br/>ground truth anchor"]
  Expert["expert_knowledge.json<br/>living knowledge base"]
  Qwen["Qwen<br/>strategist"]
  PPO["PPO<br/>explorer"]
  BC["BC retrain<br/>foundation bake-in"]

  Human --> Expert
  Expert --> Qwen
  Qwen -->|strategic features| PPO
  PPO -->|runs + outcomes| Expert
  PPO -->|decisions.jsonl| BC
  BC --> PPO
  Expert --> BC
```

| Component | Role |
|-----------|------|
| **PPO** | Explorer — finds novel action sequences through stochastic policy and offline RL |
| **Qwen** | Strategist — contextualizes discoveries, compresses run state into guidance PPO can use |
| **`expert_knowledge.json`** | Living knowledge base — Mobalytics seed + **machine-discovered** synergies and tiers |
| **BC retrain** | Each cycle bakes new knowledge into the **foundation** (imitation + expanded features) |
| **Human runs** | Ground-truth anchor — prevents drift from what actually wins in human hands |

Integration phases below are **cycle 0 → cycle 1** wiring (Qwen + features + retrain). Later cycles add **automated or semi-automated** “discover → document → retrain” tooling (not yet specified here).

### Surpassing human play (design targets)

Not by copying humans harder, but by properties humans do not have at scale:

| Advantage | Mechanism |
|-----------|-----------|
| **Consistency** | No tilt, fatigue, or session variance — every run at peak execution |
| **Perfect recall** | Full card / relic / interaction graph in the knowledge base + features |
| **No ego** | No sunk-cost attachment to a failing line; policy + knowledge update when evidence shifts |
| **Novel discovery** | PPO explores without preconception; successful weird lines get **documented**, not dismissed |

The system becomes an **extension** of human knowledge — same game, strictly larger strategy map over time.

### Philosophical endpoint

After enough cycles, the intended end state is:

- Every **viable strategy cluster** explored and represented in the knowledge base
- **Synergies and lines** no wiki or tier list captured
- A knowledge base **richer than any single human player’s mental model**
- Execution at **machine consistency** on top of that map

At that point the project is not only a strong STS2 bot — it is arguably the **most complete STS2 strategic artifact** that exists: playable policy plus documented theory discovered in play.

---

## Current state and why Qwen (near-term)

| Topic | Summary |
|--------|---------|
| **PPO ceiling (estimate)** | Floor **15–18** on current feature set and data |
| **Root cause** | PPO only sees **~195 features** encoding the *current* screen — no macro strategy |
| **Behavior at ceiling** | Policy becomes **deterministic** (local optimum): weak on synergies, archetypes, multi-floor planning |
| **Qwen role (cycle 0)** | Opens the **strategic channel** so the [knowledge loop](#the-knowledge-generation-loop) can run — not a one-off buff |

Today’s pipeline (`training/features.py` → BC → PPO) is strong at reactive combat and screen-local choices. It is not designed to represent deck identity, patch meta, or long-horizon tradeoffs unless those are explicitly featurized.

---

## Prerequisites before integration

Do **not** bolt Qwen on while PPO is still climbing — you waste integration effort and pollute ablations.

1. **PPO plateau** — two consecutive model versions each improve average floor by **less than 1** vs the prior version.
2. **Solid foundation** — plateau confirms the 195-dim policy has extracted what offline RL can from the current dataset.
3. **Dataset diversity** — plateau after broad `decisions.jsonl` collection gives Qwen varied runs to reason about (deck shapes, paths, mistakes).
4. **Fresh expert knowledge** — `cache/expert_knowledge.json` (from `tools/scrape_knowledge.py`) regenerated on the **latest balance patch** before any Qwen prompts go live.

---

## High-level architecture

**Pattern:** retrieval-augmented generation (RAG) at decision time — **no Qwen fine-tuning**. Knowledge is injected via prompts; outputs are distilled into numbers PPO already knows how to use.

```mermaid
flowchart TB
  subgraph local["Local machine"]
    Game["STS2 + MCP<br/>(up to 4 instances)"]
    Agent["sts2_agent loop"]
    LM["LM Studio<br/>Qwen 14B Q4"]
    Expert["cache/expert_knowledge.json"]
  end

  Game -->|state JSON| Agent
  Expert -->|static tiers / synergies| Agent
  Agent -->|strategic moment only| Prompt["Prompt builder"]
  Prompt -->|HTTP OpenAI-compatible| LM
  LM -->|text plan| Parser["Response parser"]
  Parser -->|Qwen feature vector| Features["195 + N features"]
  Features --> PPO["BC / PPO policy"]
  PPO -->|action| Agent
  Agent -->|POST action| Game
```

| Step | What happens |
|------|----------------|
| 1 | **LM Studio** runs locally, exposing an OpenAI-compatible HTTP API |
| 2 | At **strategic moments**, the agent builds a prompt: static knowledge + dynamic run state + a focused question |
| 3 | **Qwen** returns strategic advice as text |
| 4 | A **parser** converts text → fixed **numerical features** (concatenated with existing encoding) |
| 5 | **PPO** (same action space) chooses the final action from the expanded vector |

Combat stays on the learned policy; Qwen never blocks the hot path for every card play.

---

## When to call Qwen

| Call Qwen | Do not call Qwen |
|-----------|------------------|
| Card rewards | Every combat step |
| Rest sites | Map nodes (unless explicitly added later) |
| Shop | Hand selection / targeting |
| Boss prep / pre-boss deck checks | Fast reactive fights |

**Rationale**

- **Latency** — Qwen is too slow for 4 parallel agents polling every ~0.5s.
- **Reactivity** — combat needs millisecond-scale policy inference; PPO is already trained for that.
- **Persistence** — one Qwen response can **persist** (cached feature vector + summary) across several downstream decisions until the next strategic moment (e.g. shop → multiple purchases without re-querying).

Exact trigger list is an [open decision](#open-questions-decide-at-integration-time).

---

## Knowledge base requirements

**Current baseline** (Mobalytics scrape via `tools/scrape_knowledge.py` → `cache/expert_knowledge.json`):

| Asset | Approx. count |
|-------|----------------|
| Cards | 403 |
| Relics | 166 |
| Potions | 63 |
| Archetypes | 5 (coarse buckets) |

**Before integration**

| Requirement | Why |
|-------------|-----|
| Re-scrape after **every balance patch** | Stale tiers/synergies directly bad advice — see [Patch management](PATCH_MANAGEMENT.md) |
| **Richer archetypes** than 5 buckets | Nuanced synergy text, not single labels |
| **Contextual tiers** | e.g. “B tier normally, **S tier** in block archetype” |
| **Explicit card ↔ relic synergies** | Highest leverage for Qwen reasoning |
| Quality ∝ advice quality | Parser can only encode what the prompt describes well |

Rules bot already uses a subset via `sts2_agent/knowledge.py` (`expert_card_bonus`, archetype hints in `scorer.py`). Qwen integration should share the same source of truth, not a forked JSON.

---

## Integration phases

| Phase | Work | Exit criteria |
|-------|------|----------------|
| **1** | Install LM Studio; download **Qwen 14B Q4** (~8GB); verify local OpenAI-compatible API | `curl` / smoke test completion from repo |
| **2** | Enrich `expert_knowledge.json` — archetypes, synergies, contextual tiers | Scrape + manual curation reviewed |
| **3** | **Prompt construction** — game state + retrieved knowledge + question templates | Unit tests with fixture states |
| **4** | **Response parser** — text → `float32` Qwen feature block | Schema documented; invalid output safe fallback |
| **5** | **BC retrain** on existing `decisions.jsonl` with **195 + N** features | New `policy_net` + `model_config.json` (`feature_dim` changed) |
| **6** | **PPO retrain** from new BC warmstart (cannot continue old 195-dim checkpoints) | New `ppo_v*` with plateau tracking |
| **7** | Collect runs + evaluate | Floor progression vs `ppo_v3`-only baseline |

**Mandatory break:** Phase 5 changes `FEATURE_DIM` in `training/features.py`. Existing `.pt` weights are incompatible — same lesson as [ppo_v2 → bc_v3 retrain](MODEL_LINEAGE.md#bc-retrain-old-bc-bc_v1-vs-bc_v3).

---

## Open questions (decide at integration time)

### Triggers

- Final list of `state_type` / handler hooks that invoke Qwen
- Whether map choice (pathing) is in scope for v1 or deferred
- TTL for cached Qwen context (floors? until next shop?)

### Feature encoding

How to turn free-form text into numbers PPO can learn:

| Option | Pros | Cons |
|--------|------|------|
| Archetype score vector | Compact, stable | Loses card-specific nuance |
| Per-card priority scores (top-K) | Directly actionable at rewards/shop | Higher dim, sparse mapping |
| Binary flags (“prioritize block”, “avoid scaling”) | Simple, interpretable | Coarse |
| Embedding of summary sentence | Rich | Needs dim reduction; harder to debug |

**Decision needed:** `N` = number of new dimensions (drives BC dataset rebuild and model size).

### Infrastructure (4 parallel agents)

| Option | Notes |
|--------|--------|
| **One shared Qwen** | Serialize strategic calls; queue across agents |
| **One Qwen per agent** | 4 × ~8GB — not viable on **8GB VRAM** with game clients |
| **CPU offload for Qwen** | Frees VRAM for game; slower but may be acceptable at strategic cadence |

Likely default: **single shared instance** + request queue + aggressive caching of last strategic context per run.

### Latency vs data collection

- Strategic calls only → minimal impact on steps/sec
- Log Qwen prompt hash + parsed features into `decisions.jsonl` for offline replay without re-inference
- Optional: async pre-fetch Qwen before predicted strategic screen (e.g. after combat when rewards next)

---

## Risks

| Risk | Mitigation |
|------|------------|
| **VRAM contention** — Qwen 14B Q4 ~8GB + game instances | CPU offload; strategic-only calls; single shared model |
| **Knowledge staleness** | Patch checklist: scrape → validate → bump `agent_version` / doc date |
| **Prompt quality** | Template review; golden-run regression; human spot-check logs |
| **Feature dim change** | Plan explicit `bc_v4` / `ppo_v4` lineage; no warmstart from 195-dim weights |
| **Qwen bad advice** | Fallback to zeros / last good cache; rules layer still available |
| **Evaluation confound** | Hold dataset split constant; A/B `agent_version` tags in dashboard |

---

## Success metrics (phase 7)

- **Primary:** median / p75 floor vs `ppo_v3` at same run count
- **Secondary:** win rate, boss reach rate, deck quality proxies (card_reward picks vs expert tier)
- **Guardrails:** no increase in stuck loops / invalid actions; strategic call latency p95 documented

---

## Suggested timeline (logical, not calendar)

**First integration cycle** (static knowledge + Qwen features):

```mermaid
flowchart LR
  A["ppo_v3 plateau"] --> B["Phase 1–2<br/>LM + knowledge"]
  B --> C["Phase 3–4<br/>Prompt + parser"]
  C --> D["Phase 5–6<br/>bc_v4 + ppo_v4"]
  D --> E["Phase 7<br/>Evaluate"]
```

Until **A** is satisfied, prefer: more diverse agent runs, PPO hyperparameter sweeps, and knowledge scrape updates — not Qwen integration.

**Ongoing cycles** (after phase 7):

```mermaid
flowchart LR
  Run["Agent runs"] --> Mine["Mine high-floor / novel decks"]
  Mine --> Write["Update expert_knowledge.json"]
  Write --> Retrain["BC + PPO retrain"]
  Retrain --> Run
```

Each lap is one turn of the [self-improving system](#long-term-vision--self-improving-system).

---

## Changelog

| Date | Note |
|------|------|
| 2026-05-18 | Initial roadmap (post–`ppo_v3`, pre-Qwen) |
| 2026-05-18 | Added self-improving system vision, knowledge loop, five components |
