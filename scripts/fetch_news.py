"""Fetch AI/LLM news from Hacker News API and write to src/data/news.json"""
import json
import urllib.request
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

KEYWORDS = [
    'llm', 'large language model', 'gpt', 'claude', 'gemini', 'openai',
    'anthropic', 'ai agent', 'transformer', 'fine-tun', 'rag',
    'embedding', 'vector', 'chatbot', 'diffusion', 'stable diffusion',
    'midjourney', 'copilot', 'machine learning', 'deep learning',
    'neural network', 'artificial intelligence', 'ai model',
    'prompt engineer', 'inference', 'quantiz', 'gguf', 'ollama',
    'langchain', 'llamaindex', 'hugging face', 'mistral', 'llama',
    'deepseek', 'qwen', 'ai infra', 'mlops', 'generative ai',
]

HN_TOP = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM = "https://hacker-news.firebaseio.com/v0/item/{}.json"

def is_ai_related(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in KEYWORDS)

def fetch_json(url: str):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def main():
    print("Fetching top stories from Hacker News...")
    top_ids = fetch_json(HN_TOP)[:100]

    results = []
    for sid in top_ids:
        try:
            item = fetch_json(HN_ITEM.format(sid))
        except Exception:
            continue
        if not item or item.get('type') != 'story' or not item.get('title'):
            continue
        if not is_ai_related(item['title']):
            continue

        ts = item.get('time', 0)
        dt = datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=8)))

        results.append({
            'title': item['title'],
            'url': item.get('url', f"https://news.ycombinator.com/item?id={sid}"),
            'source': 'Hacker News',
            'date': dt.strftime('%Y-%m-%d'),
            'points': item.get('score', 0),
            'comments': item.get('descendants', 0),
        })

    results.sort(key=lambda x: x['points'], reverse=True)
    results = results[:30]

    out_path = Path(__file__).resolve().parent.parent / 'src' / 'data' / 'news.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. {len(results)} AI-related articles written to {out_path}")

if __name__ == '__main__':
    main()
