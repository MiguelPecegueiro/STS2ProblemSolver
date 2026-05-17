import requests

text = requests.get(
    "https://raw.githubusercontent.com/Gennadiyev/STS2MCP/main/docs/raw-full.md",
    timeout=60,
).text
for marker in ("### `rewards`", "potion", "claim_reward", "potion_full", "discard_potion"):
    idx = 0
    shown = 0
    while shown < 2:
        i = text.lower().find(marker.lower(), idx)
        if i < 0:
            break
        print(f"--- {marker} @ {i} ---")
        print(text[max(0, i - 40) : i + 900])
        print()
        idx = i + len(marker)
        shown += 1
