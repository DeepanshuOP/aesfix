import json
import os
import hashlib

CACHE_FILE = "cache/llm_cache.json"

def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def _save_cache(cache: dict):
    os.makedirs("cache", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def get_cached(text: str) -> dict | None:
    cache = _load_cache()
    key = hashlib.md5(text.strip().lower().encode()).hexdigest()
    return cache.get(key)

def set_cached(text: str, result: dict):
    cache = _load_cache()
    key = hashlib.md5(text.strip().lower().encode()).hexdigest()
    cache[key] = result
    _save_cache(cache)
