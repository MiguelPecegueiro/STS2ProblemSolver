import requests

text = requests.get(
    "https://raw.githubusercontent.com/Gennadiyev/STS2MCP/main/docs/raw-full.md",
    timeout=60,
).text
start = text.find("### `card_select`")
print(text[start : start + 3500])
print("\n---ACTIONS---\n")
start = text.find("### `select_card`")
print(text[start : start + 2000])
