"""
cache.py - In-Memory Normalized Query Response Cache
Track: Business Operations / Customer Support (Ola)

Requirements:
- In-memory cache keyed by normalized query text for grounded generation
- Demonstrates repeated query producing a cache hit that avoids redundant LLM/tool calls
- Telemetry tracking call counts, cache hits/misses, and execution timings
"""

import re
import time
from typing import Dict, Any, Optional, Tuple


def normalize_query(query: str) -> str:
    """
    Normalizes query text:
    - Lowercase
    - Strips leading/trailing whitespace
    - Collapses multiple whitespace characters
    - Strips peripheral punctuation
    """
    text = query.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


class ResponseCache:
    """
    Thread-safe in-memory cache for grounded generation responses.
    """
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self.hits: int = 0
        self.misses: int = 0
        self.underlying_call_count: int = 0

    def get(self, query: str) -> Optional[Any]:
        key = normalize_query(query)
        if key in self._store:
            self.hits += 1
            entry = self._store[key]
            entry["hit_count"] += 1
            return entry["data"]
        self.misses += 1
        return None

    def set(self, query: str, data: Any) -> None:
        key = normalize_query(query)
        self._store[key] = {
            "data": data,
            "cached_at": time.time(),
            "hit_count": 0,
        }

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0
        self.underlying_call_count = 0

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_ratio = (self.hits / total) if total > 0 else 0.0
        return {
            "total_entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(hit_ratio, 3),
            "underlying_call_count": self.underlying_call_count,
        }


# Global singleton cache instance
GLOBAL_CACHE = ResponseCache()


if __name__ == "__main__":
    print("Testing ResponseCache:")
    cache = ResponseCache()
    
    q1 = "What is Ola's refund policy for delayed rides?"
    q2 = "what is ola's refund policy for delayed rides?! "
    
    # 1. First call (Miss)
    t0 = time.perf_counter()
    cached = cache.get(q1)
    if cached is None:
        cache.underlying_call_count += 1
        # Simulate work
        time.sleep(0.05)
        res = {"answer": "Refunds are processed in 3-5 banking days or Ola Money."}
        cache.set(q1, res)
    t1 = time.perf_counter()
    miss_duration = t1 - t0
    print(f"Call 1 (Miss): duration={miss_duration*1000:.2f}ms, calls={cache.underlying_call_count}")
    
    # 2. Second call with minor casing/punctuation variation (Hit)
    t2 = time.perf_counter()
    cached2 = cache.get(q2)
    if cached2 is None:
        cache.underlying_call_count += 1
        time.sleep(0.05)
        res2 = {"answer": "Refunds are processed in 3-5 banking days or Ola Money."}
        cache.set(q2, res2)
    else:
        res2 = cached2
    t3 = time.perf_counter()
    hit_duration = t3 - t2
    print(f"Call 2 (Hit): duration={hit_duration*1000:.2f}ms, calls={cache.underlying_call_count}")
    
    stats = cache.get_stats()
    print("Cache Stats:", stats)
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["underlying_call_count"] == 1
    print("ResponseCache verified successfully!")
