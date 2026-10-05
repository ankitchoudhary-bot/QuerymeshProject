"""Redis client / cache factory — resolves from `Settings`.

Works for real Redis and for Upstash's Redis-compatible endpoint (both
speak the same protocol `redis.asyncio.Redis.from_url` expects).
"""

from __future__ import annotations

from querymesh.cache.errors import CacheError
from querymesh.cache.response_cache import ResponseCache
from querymesh.config import Settings, get_settings
from querymesh.observability.tracing import Tracer


def build_redis_client(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.redis_url:
        raise CacheError("REDIS_URL is not configured")

    import redis.asyncio as redis

    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def get_response_cache(settings: Settings | None = None, tracer: Tracer | None = None) -> ResponseCache:
    settings = settings or get_settings()
    client = build_redis_client(settings)
    return ResponseCache(
        client=client,
        ttl_seconds=settings.redis_ttl_seconds,
        tracer=tracer,
    )
