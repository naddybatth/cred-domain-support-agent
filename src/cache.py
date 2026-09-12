"""Task 16 - in-memory response cache keyed by normalized query text."""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def normalize_query(query: str) -> str:
    """Normalisation is what makes 'What is the EMI formula?' and
    '  what is the EMI formula  ' the same cache key."""
    return _WS.sub(" ", _PUNCT.sub("", query.lower())).strip()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    saved_calls: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "saved_llm_or_tool_calls": self.saved_calls,
            "hit_rate": self.hit_rate,
        }


@dataclass
class ResponseCache:
    max_size: int = 256
    _store: Dict[str, Any] = field(default_factory=dict)
    _timing: Dict[str, float] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_compute(
        self, query: str, compute: Callable[[], Any]
    ) -> Tuple[Any, bool, float]:
        """Returns (value, was_cache_hit, elapsed_seconds)."""
        key = normalize_query(query)
        with self._lock:
            if key in self._store:
                start = time.perf_counter()
                value = self._store[key]
                self.stats.hits += 1
                self.stats.saved_calls += 1
                return value, True, time.perf_counter() - start

        start = time.perf_counter()
        value = compute()
        elapsed = time.perf_counter() - start
        with self._lock:
            if len(self._store) >= self.max_size:
                oldest = next(iter(self._store))
                self._store.pop(oldest, None)
                self._timing.pop(oldest, None)
            self._store[key] = value
            self._timing[key] = elapsed
            self.stats.misses += 1
        return value, False, elapsed

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._timing.clear()
            self.stats = CacheStats()


GENERATION_CACHE = ResponseCache()
