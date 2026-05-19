# Local monster database (Spire Codex)

Sectioned monster snapshots for offline use, combat-sim planning, and the dashboard enemy compendium.

## Layout

```
data/monsters/
  index.json
  by_type/
    normal.json
    elite.json
    boss.json
    other.json
```

## Update

```powershell
py tools/import_codex_monsters.py --force
```

Without `--force`, skips download if `index.json` is younger than 24 hours.

## Code

- `sts2_agent/codex_monsters.py` — fetch, classify, write, load helpers
- `tools/import_codex_monsters.py` — CLI
- `dashboard/monster_compendium.py` — Streamlit browser
