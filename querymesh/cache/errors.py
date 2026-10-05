"""Cache-layer errors."""

from __future__ import annotations

from querymesh.errors import QueryMeshError


class CacheError(QueryMeshError):
    """Base error for cache-layer failures. Retryable by default — most
    cache failures (Redis connection blip) are transient; a cache miss on
    retry is harmless (falls through to computing the answer fresh).
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)
