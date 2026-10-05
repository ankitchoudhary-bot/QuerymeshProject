"""Embeddings abstraction.

Rule: the rest of the application depends on the `Embedder` protocol, not
on Jina (or any provider) directly — mirrors the `LLM` abstraction from
Phase 1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from querymesh.errors import QueryMeshError


class EmbeddingError(QueryMeshError):
    """Base error for embedding-layer failures. Retryable by default —
    most embedding errors (rate limits, transient network issues) are.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class EmbeddingDimensionError(EmbeddingError):
    """Raised when returned embeddings don't have the expected dimension.
    Never retryable — a dimension mismatch means misconfiguration (wrong
    model/index), not a transient failure; retrying gets the same result.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


@runtime_checkable
class Embedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document/passage texts."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        ...
