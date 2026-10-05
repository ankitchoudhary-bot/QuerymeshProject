"""Exact-match response caching.

Caches the final answer + citations + metadata for a query, keyed by
`make_cache_key()`. Rule: "Do not cache failed/incomplete responses" —
`set()` silently skips caching when the answer is empty rather than
storing a blank result, so a transient failure never poisons the cache.

`get`/`set` each open a "cache" span with `cache_hit`/`cache_miss` so the
observability layer can track cache effectiveness.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from querymesh.cache.keys import make_cache_key
from querymesh.observability.logging import get_logger
from querymesh.observability.tracing import NoOpTracer, Tracer
from querymesh.retrieval.models import RetrievedDocument
logger = get_logger(__name__)

_DEFAULT_TTL_SECONDS = 3600


@runtime_checkable
class RedisClientProtocol(Protocol):
    """The subset of `redis.asyncio.Redis` this cache actually uses."""

    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: str, ex: int | None = None) -> Any: ...
    async def delete(self, key: str) -> Any: ...
    async def exists(self, key: str) -> Any: ...
    async def close(self) -> Any: ...


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    if hasattr(obj, "value"):
        return obj.value

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

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

    async def set(self, query: str, answer: str, citations: list[dict], retrieved_documents: list[RetrievedDocument] | None = None, evidence: list[dict] | None = None, metadata: dict | None = None, ) -> bool:
        """Cache a completed answer. Returns False (no-op) for an empty
        answer, since an empty answer means the request didn't actually
        succeed — never cache that.
        """
        if not answer or not answer.strip():
            logger.warning("Skipping cache write for empty answer (query=%r)", query)
            return False

        key = self._key(query)
        payload = json.dumps({"answer": answer, "citations": citations, "retrieved_documents": retrieved_documents or [], "evidence": evidence, "metadata": metadata or {}}, default=_json_default)
        await self._client.set(key, payload, ex=self.ttl_seconds)
        return True

    async def delete(self, query: str) -> bool:
        key = self._key(query)
        deleted = await self._client.delete(key)
        return bool(deleted)

    async def exists(self, query: str) -> bool:
        key = self._key(query)
        return bool(await self._client.exists(key))
