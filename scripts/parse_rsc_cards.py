"""Try to extract card stats from sts2replays RSC inline payloads."""
import json
import re
from pathlib import Path

text = Path("scripts/full_stats.html").read_text(encoding="utf-8")

# Look for repeated card-stat shaped objects in escaped JSON
pattern = re.compile(
    r'\{\\"name\\":\\"([^"\\]+)\\",\\"winRate\\":([\d.]+),\\"pickRate\\":([\d.]+)'
)
matches = pattern.findall(text)
print("pattern1 matches", len(matches), matches[:5])

# Broader: any winRate number near a card name
pattern2 = re.compile(r'\\"winRate\\":([\d.]+)')
rates = pattern2.findall(text)
print("winRate values", len(rates), rates[:10])

# Search for cardStats array marker
for marker in ["cardStats", "cardsStats", "allCards", "card_popularity"]:
    print(marker, text.find(marker))
