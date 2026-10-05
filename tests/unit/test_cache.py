"""Phase 19 tests: cache key normalization/SHA256, ResponseCache
(get/set/delete/exists, TTL, no-cache-on-empty-answer, cache_hit/miss
tracing), and the Redis client/cache factory — via an in-memory fake
Redis double, no live Redis/Upstash server needed.
"""

from __future__ import annotations

import hashlib

import pytest

from querymesh.cache import CacheError, ResponseCache, build_redis_client, get_response_cache, make_cache_key, normalize_query
from querymesh.config import Settings


# --- Key construction ----------------------------------------------------


@pytest.mark.unit
def test_normalize_query_collapses_whitespace_and_case() -> None:
    assert normalize_query("  What IS   Attention?  ") == "what is attention?"


@pytest.mark.unit
def test_make_cache_key_is_sha256_of_normalized_query() -> None:
    key = make_cache_key("What is attention?")
    expected_digest = hashlib.sha256("what is attention?".encode("utf-8")).hexdigest()
    assert key == f"querymesh:answer:{expected_digest}"


@pytest.mark.unit
def test_make_cache_key_same_for_trivially_different_phrasing() -> None:
    key1 = make_cache_key("What is attention?")
    key2 = make_cache_key("  what   IS attention?")
    assert key1 == key2


@pytest.mark.unit
def test_make_cache_key_custom_prefix() -> None:
    key = make_cache_key("q", prefix="custom:prefix")
    assert key.startswith("custom:prefix:")


# --- ResponseCache --------------------------------------------------------


class _FakeRedisClient:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, int | None]] = []
        self.closed = False

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self._store[key] = value
        self.set_calls.append((key, value, ex))
        return True

    async def delete(self, key: str):
        existed = key in self._store
        self._store.pop(key, None)
        return 1 if existed else 0

    async def exists(self, key: str):
        return 1 if key in self._store else 0

    async def close(self):
        self.closed = True


@pytest.mark.unit
async def test_set_then_get_roundtrip() -> None:
    client = _FakeRedisClient()
    cache = ResponseCache(client=client)  # type: ignore[arg-type]

    ok = await cache.set(
        "What is attention?",
        answer="Attention is a mechanism...",
        citations=[{"chunk_id": "a:chunk:0000"}],
        metadata={"model": "test"},
    )
    result = await cache.get("What is attention?")

    assert ok is True
    assert result["answer"] == "Attention is a mechanism..."
    assert result["citations"] == [{"chunk_id": "a:chunk:0000"}]
    assert result["metadata"] == {"model": "test"}


@pytest.mark.unit
async def test_get_returns_none_on_miss() -> None:
    client = _FakeRedisClient()
    cache = ResponseCache(client=client)  # type: ignore[arg-type]

    result = await cache.get("never cached")
    assert result is None


@pytest.mark.unit
async def test_set_uses_configured_ttl() -> None:
    client = _FakeRedisClient()
    cache = ResponseCache(client=client, ttl_seconds=120)

    await cache.set("q", answer="a", citations=[])

    assert len(client.set_calls) == 1
    _, _, ex = client.set_calls[0]
    assert ex == 120


@pytest.mark.unit
async def test_set_skips_empty_answer() -> None:
    client = _FakeRedisClient()
    cache = ResponseCache(client=client)  # type: ignore[arg-type]

    ok = await cache.set("q", answer="", citations=[])
    ok_whitespace = await cache.set("q2", answer="   ", citations=[])

    assert ok is False
    assert ok_whitespace is False
    assert client.set_calls == []


@pytest.mark.unit
async def test_delete_removes_entry() -> None:
    client = _FakeRedisClient()
    cache = ResponseCache(client=client)  # type: ignore[arg-type]
    await cache.set("q", answer="a", citations=[])

    deleted = await cache.delete("q")
    still_there = await cache.exists("q")

    assert deleted is True
    assert still_there is False


@pytest.mark.unit
async def test_delete_nonexistent_returns_false() -> None:
    client = _FakeRedisClient()
    cache = ResponseCache(client=client)  # type: ignore[arg-type]

    assert await cache.delete("never set") is False


@pytest.mark.unit
async def test_exists_true_after_set() -> None:
    client = _FakeRedisClient()
    cache = ResponseCache(client=client)  # type: ignore[arg-type]
    await cache.set("q", answer="a", citations=[])

    assert await cache.exists("q") is True


@pytest.mark.unit
async def test_different_queries_use_different_keys() -> None:
    client = _FakeRedisClient()
    cache = ResponseCache(client=client)  # type: ignore[arg-type]

    await cache.set("query one", answer="answer one", citations=[])
    await cache.set("query two", answer="answer two", citations=[])

    r1 = await cache.get("query one")
    r2 = await cache.get("query two")

    assert r1["answer"] == "answer one"
    assert r2["answer"] == "answer two"


# --- cache_hit / cache_miss tracing --------------------------------------


class _SpyTracer:
    def __init__(self) -> None:
        self.spans: list[dict] = []

    def span(self, name: str, **attributes):
        import contextlib

        @contextlib.contextmanager
        def _cm():
            self.spans.append({"name": name, **attributes})
            yield

        return _cm()


@pytest.mark.unit
async def test_get_records_cache_miss_span() -> None:
    client = _FakeRedisClient()
    spy = _SpyTracer()
    cache = ResponseCache(client=client, tracer=spy)  # type: ignore[arg-type]

    await cache.get("not cached")

    assert spy.spans[-1]["name"] == "cache"
    assert spy.spans[-1]["cache_hit"] is False


@pytest.mark.unit
async def test_get_records_cache_hit_span() -> None:
    client = _FakeRedisClient()
    spy = _SpyTracer()
    cache = ResponseCache(client=client, tracer=spy)  # type: ignore[arg-type]
    await cache.set("q", answer="a", citations=[])

    await cache.get("q")

    assert spy.spans[-1]["name"] == "cache"
    assert spy.spans[-1]["cache_hit"] is True


# --- Factory ------------------------------------------------------------


@pytest.mark.unit
def test_build_redis_client_raises_without_url() -> None:
    settings = Settings(REDIS_URL=None)
    with pytest.raises(CacheError):
        build_redis_client(settings)


@pytest.mark.unit
def test_build_redis_client_constructs_real_client_when_configured() -> None:
    settings = Settings(REDIS_URL="redis://localhost:6379/0")
    client = build_redis_client(settings)

    import redis.asyncio as redis

    assert isinstance(client, redis.Redis)


@pytest.mark.unit
def test_get_response_cache_uses_configured_ttl() -> None:
    settings = Settings(REDIS_URL="redis://localhost:6379/0", REDIS_TTL_SECONDS=999)
    cache = get_response_cache(settings)

    assert isinstance(cache, ResponseCache)
    assert cache.ttl_seconds == 999
