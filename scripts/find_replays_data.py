import json
import re
from pathlib import Path

import requests

text = requests.get(
    "https://sts2replays.com/stats",
    headers={"User-Agent": "Mozilla/5.0 STS2Agent/0.1"},
    timeout=60,
).text
Path("scripts/full_stats.html").write_text(text, encoding="utf-8")

# __NEXT_DATA__
m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', text, re.S)
if m:
    data = json.loads(m.group(1))
    Path("scripts/next_data.json").write_text(
        json.dumps(data, indent=2)[:500000], encoding="utf-8"
    )
    print("NEXT_DATA keys", list(data.keys()))
    props = data.get("props", {})
    print("pageProps keys", list(props.get("pageProps", {}).keys())[:20])

# search urls
for pat in [r"https://[^\"'\s]+", r'"/_next/data/[^"]+"']:
    found = sorted(set(re.findall(pat, text)))
    print(pat, "count", len(found))
    for u in found[:15]:
        if "card" in u.lower() or "stat" in u.lower() or "api" in u.lower():
            print(" ", u)
