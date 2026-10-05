"""Deterministic fake embedder for tests and offline development.

Not a mock — a real, first-class implementation of `Embedder`. Uses a
seeded hash of each text to produce a deterministic, fixed-dimension
vector, so the same text always embeds to the same vector (useful for
testing vector search and RRF later without a real model).
"""

from __future__ import annotations

import hashlib

from querymesh.embeddings.base import Embedder  # noqa: F401  (documents the contract)


class FakeEmbedder:
    """Deterministic hash-based embedder implementing the `Embedder` protocol."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _embed_text(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat the digest bytes to fill `dimension`, normalize to [-1, 1].
        values = [digest[i % len(digest)] for i in range(self.dimension)]
        return [(v / 127.5) - 1.0 for v in values]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        return self._embed_text(query)
