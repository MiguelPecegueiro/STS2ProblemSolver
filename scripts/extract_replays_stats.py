import json
import re
from pathlib import Path

import requests

r = requests.get(
    "https://sts2replays.com/stats",
    headers={"User-Agent": "Mozilla/5.0 STS2Agent/0.1"},
    timeout=60,
)
text = r.text
Path("scripts/stats_page_sample.txt").write_text(text[62000:65000], encoding="utf-8")

# Next.js flight / RSC payloads often embed JSON after pickRate
for key in ("pickRate", "winRate", "cardName"):
    idx = text.find(key)
    print(key, idx)

# Try to find JSON array of card objects
m = re.search(r'\{"cards":\s*\[', text)
print("cards array", m.start() if m else None)

# self.__next_f.push pattern
chunks = re.findall(r'self\.__next_f\.push\(\[1,"((?:\\.|[^"\\])*)"\]\)', text)
print("next_f chunks", len(chunks))
