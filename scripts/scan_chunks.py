import re
from pathlib import Path

import requests

text = Path("scripts/full_stats.html").read_text(encoding="utf-8")
chunks = re.findall(r'src="(/_next/static/chunks/[^"]+)"', text)
print("chunks", len(chunks))

for rel in chunks:
    url = "https://sts2replays.com" + rel.split("?")[0]
    try:
        js = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    except Exception:
        continue
    if "winRate" in js and ("card" in js.lower() or "Card" in js):
        hits = [kw for kw in ["api/", "fetch(", "cardStats", "/stats", "winRate"] if kw in js]
        if hits:
            print(rel, hits)
            idx = js.find("api/")
            if idx >= 0:
                print("  snippet:", js[idx : idx + 120])
