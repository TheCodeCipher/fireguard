import asyncio

from fireguard.services.cache import AsyncTTLCache


def test_ttl_cache_loads_once_and_reports_hit():
    async def run():
        cache = AsyncTTLCache[str]()
        calls = 0

        async def loader():
            nonlocal calls
            calls += 1
            return "observations"

        first = await cache.get_or_load("source:bbox:date", 60, loader)
        second = await cache.get_or_load("source:bbox:date", 60, loader)
        assert first.hit is False
        assert second.hit is True
        assert first.value == second.value == "observations"
        assert calls == 1

    asyncio.run(run())


def test_ttl_cache_does_not_reuse_other_query_key():
    async def run():
        cache = AsyncTTLCache[str]()
        a = await cache.get_or_load("VIIRS_SNPP_SP|bbox-A", 60, lambda: _value("A"))
        b = await cache.get_or_load("VIIRS_NOAA20_SP|bbox-A", 60, lambda: _value("B"))
        assert a.value == "A"
        assert b.value == "B"
        assert b.hit is False

    asyncio.run(run())


async def _value(value: str) -> str:
    return value
