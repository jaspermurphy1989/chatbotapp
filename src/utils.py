import hashlib
from pathlib import Path
from typing import Any, Dict, List
import json
from config import config

def get_cache_key(query: str) -> str:
    """Generate a consistent cache key for a query."""
    return hashlib.md5(query.encode()).hexdigest()

def save_to_cache(key: str, data: Any):
    """Save data to cache with the given key."""
    cache_file = config.CACHE_DIR / f"{key}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f)

def load_from_cache(key: str) -> Any:
    """Load data from cache if exists."""
    cache_file = config.CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None

def clear_cache():
    """Clear all cache files."""
    for file in config.CACHE_DIR.glob("*.json"):
        file.unlink()
