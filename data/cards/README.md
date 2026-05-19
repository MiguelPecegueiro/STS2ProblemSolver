# Local card database (Spire Codex)

Sectioned card snapshots for offline use, combat-sim planning, and tooling.

## Layout

```
data/cards/
  index.json              # manifest: counts, paths, fetch time
  by_color/
    ironclad.json         # 87 cards
    silent.json
    defect.json
    necbinder.json
    regent.json
    colorless.json
    token.json
    curse.json
    status.json           # Codex pool (burns, etc.)
    event.json            # event-granted cards
    quest.json
```

Each section file:

```json
{
  "section": "ironclad",
  "label": "Ironclad",
  "fetched_at": "...",
  "source": "https://spire-codex.com/api/cards",
  "count": 87,
  "cards": [ /* full Codex card objects + section, section_label */ ]
}
```

## Update

```powershell
py tools/import_codex_cards.py --force
```

Without `--force`, skips download if `index.json` is younger than 24 hours.

## Code

- `sts2_agent/codex_cards.py` — fetch, classify, write, load helpers
- `tools/import_codex_cards.py` — CLI

`cache/knowledge.json` remains the agent’s full Codex bundle (relics, monsters, …). This tree is **cards only**, organized by color for humans and future sim import.
