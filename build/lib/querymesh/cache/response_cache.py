"""Exact-match response caching.

Caches the final answer + citations + metadata for a query, keyed by
`make_cache_key()`. Rule: "Do not cache failed/incomplete responses" —
`set()` silently skips caching when the answer is empty rather than
storing a blank result, so a transient failure never poisons the cache.

`get`/`set` each open a "cache" span with `cache_hit`/`cache_miss` so the
observability layer (Phase 18) can track cache effectiveness — this is
the wiring Phase 18 left as a known limitation.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from querymesh.cache.keys import make_cache_key
from querymesh.observability.logging import get_logger
from querymesh.observability.tracing import NoOpTracer, Tracer

logger = get_logger(__name__)

_DEFAULT_TTL_SECONDS = 3600


@runtime_checkable
class RedisClientProtocol(Protocol):
    """The subset of `redis.asyncio.Redis` this cache actually uses.

    Lets tests supply an in-memory fake without a real Redis server,
    exactly like `OpenSearchClientProtocol` in Phase 5.
    """

    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> Any: ...
    async def delete(self, key: str) -> Any: ...
    async def exists(self, key: str) -> Any: ...
    async def close(self) -> Any: ...


class ResponseCache:
    """Redis-backed cache for final research answers."""

    def __init__(
        self,
        client: RedisClientProtocol,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
        key_prefix: str = "querymesh:answer",
        tracer: Tracer | None = None,
    ) -> None:
        self._client = client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.tracer = tracer or NoOpTracer()

    def _key(self, query: str) -> str:
        return make_cache_key(query, prefix=self.key_prefix)

    async def get(self, query: str) -> dict | None:
        key = self._key(query)
        raw = await self._client.get(key)

        if raw is None:
            with self.tracer.span("cache", cache_hit=False, key=key):
                pass
            return None

        with self.tracer.span("cache", cache_hit=True, key=key):
            pass
        return json.loads(raw)

    async def set(
        self,
        query: str,
        answer: str,
        citations: list[dict],
        metadata: dict | None = None,
    ) -> bool:
        """Cache a completed answer. Returns False (no-op) for an empty
        answer, since an empty answer means the request didn't actually
        succeed — never cache that.
        """
        if not answer or not answer.strip():
            logger.warning("Skipping cache write for empty answer (query=%r)", query)
            return False

        key = self._key(query)
        payload = json.dumps({"answer": answer, "citations": citations, "metadata": metadata or {}})
        await self._client.set(key, payload, ex=self.ttl_seconds)
        return True

    async def delete(self, query: str) -> bool:
        key = self._key(query)
        deleted = await self._client.delete(key)
        return bool(deleted)

    async def exists(self, query: str) -> bool:
        key = self._key(query)
        return bool(await self._client.exists(key))
