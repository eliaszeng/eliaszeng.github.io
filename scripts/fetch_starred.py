"""Fetch starred repos from GitHub API and write to src/data/starred.json"""
import json
import os
import urllib.request
from pathlib import Path

USERNAME = "eliaszeng"
TOKEN = os.environ.get("GITHUB_TOKEN", "")

def fetch_page(page: int = 1) -> list:
    url = f"https://api.github.com/users/{USERNAME}/starred?per_page=100&page={page}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "fetch-starred-script",
    }
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def main():
    print(f"Fetching starred repos for {USERNAME}...")
    repos = []
    page = 1
    while True:
        batch = fetch_page(page)
        if not batch:
            break
        repos.extend(batch)
        page += 1

    results = []
    for repo in repos:
        results.append({
            "name": repo["full_name"],
            "description": repo.get("description") or "",
            "language": repo.get("language") or "",
            "stars": repo.get("stargazers_count", 0),
            "url": repo["html_url"],
        })

    out_path = Path(__file__).resolve().parent.parent / "src" / "data" / "starred.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(results)} starred repos written to {out_path}")

if __name__ == "__main__":
    main()
