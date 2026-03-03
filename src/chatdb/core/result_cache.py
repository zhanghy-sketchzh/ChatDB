"""
查询结果缓存

基于内存的 LRU + TTL 缓存，避免完全相同的查询重复执行。
"""

import hashlib
import time
from typing import Any


class ResultCache:
    """查询结果内存缓存"""

    def __init__(self, max_size: int = 50, ttl: float = 300.0):
        self._cache: dict[str, dict[str, Any]] = {}
        self._max_size = max_size
        self._ttl = ttl

    @staticmethod
    def _make_key(session_id: str | None, query: str) -> str:
        raw = f"{session_id or ''}:{query.strip().lower()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, session_id: str | None, query: str) -> dict[str, Any] | None:
        """获取缓存结果，未命中或过期返回 None"""
        key = self._make_key(session_id, query)
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > self._ttl:
            del self._cache[key]
            return None
        return entry["result"]

    def put(self, session_id: str | None, query: str, result: dict[str, Any]) -> None:
        """缓存成功的查询结果"""
        if not result.get("success"):
            return
        key = self._make_key(session_id, query)
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
        self._cache[key] = {"result": result, "timestamp": time.time()}

    def invalidate(self, session_id: str | None = None) -> int:
        """清除缓存，返回被清除的条目数"""
        count = len(self._cache)
        self._cache.clear()
        return count
