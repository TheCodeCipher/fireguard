import asyncio
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CacheResult(Generic[T]):
    value: T
    hit: bool
    expires_at_epoch: float


class AsyncTTLCache(Generic[T]):
    """Small process-local TTL cache for the single-process v1 API.

    A shared deployment should replace this adapter with a shared cache only if
    measured traffic requires it. Cache keys must include all data-affecting filters.
    """

    def __init__(self) -> None:
        self._items: dict[str, tuple[float, T]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CacheResult[T] | None:
        async with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= time.monotonic():
                self._items.pop(key, None)
                return None
            return CacheResult(value=value, hit=True, expires_at_epoch=expires_at)

    async def set(self, key: str, value: T, ttl_seconds: int) -> float:
        expires_at = time.monotonic() + max(ttl_seconds, 0)
        if ttl_seconds <= 0:
            return expires_at
        async with self._lock:
            self._items[key] = (expires_at, value)
        return expires_at

    async def get_or_load(self, key: str, ttl_seconds: int, loader):
        existing = await self.get(key)
        if existing is not None:
            return existing
        value = await loader()
        expires_at = await self.set(key, value, ttl_seconds)
        return CacheResult(value=value, hit=False, expires_at_epoch=expires_at)

    async def clear(self) -> None:
        async with self._lock:
            self._items.clear()
