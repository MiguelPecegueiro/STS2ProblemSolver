import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) STS2Agent/0.1",
    "Accept": "application/json",
}

paths = [
    "/api/cards",
    "/api/cards/stats",
    "/api/card-stats",
    "/api/public/cards",
    "/api/public/stats/cards",
    "/api/stats/cards",
    "/api/v1/cards/stats",
    "/api/v1/stats/cards",
    "/api/analytics/cards",
]

for path in paths:
    url = "https://sts2replays.com" + path
    r = requests.get(url, headers=HEADERS, timeout=15)
    ct = r.headers.get("content-type", "")
    body = r.text[:120].replace("\n", " ")
    print(f"{r.status_code} {path} {ct} {body}")
