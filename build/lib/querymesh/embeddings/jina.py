"""Jina AI embeddings client.

Implements `Embedder` against Jina's `/v1/embeddings` endpoint. Never
takes an API key as a literal — always sourced from `Settings` /
`JINA_API_KEY`, and the caller is responsible for keeping it out of logs
(this module never logs the key or request bodies containing it).
"""

from __future__ import annotations

import asyncio

import httpx

from querymesh.embeddings.base import EmbeddingDimensionError, EmbeddingError
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "https://api.jina.ai/v1/embeddings"
_DEFAULT_BATCH_SIZE = 64
_REQUEST_TIMEOUT_SECONDS = 30.0
_RETRY_BASE_DELAY_SECONDS = 0.5


class JinaEmbedder:
    """Embedder backed by the Jina AI embeddings API.

    `task` follows Jina's convention of distinguishing how a text will be
    used (`retrieval.passage` for documents being indexed, `retrieval.query`
    for search queries) — this improves retrieval quality for models that
    support it and is set deterministically per call, not left ambiguous.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = _DEFAULT_BASE_URL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        expected_dimension: int | None = None,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise EmbeddingError("JinaEmbedder requires a non-empty api_key")
        if not model:
            raise EmbeddingError("JinaEmbedder requires a model name")
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be positive")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size
        self.expected_dimension = expected_dimension
        self.max_retries = max_retries

        self._transport = transport
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT_SECONDS, transport=self._transport
            )
        return self._client

    async def _embed_batch(self, texts: list[str], task: str) -> list[list[float]]:
        client = await self._get_client()
        payload = {"model": self.model, "input": texts, "task": task}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(self.base_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                break
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "Jina embeddings request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))
        else:
            raise EmbeddingError(
                f"Jina embeddings request failed after {self.max_retries + 1} attempts: {last_exc}"
            ) from last_exc

        try:
            items = sorted(data["data"], key=lambda item: item["index"])
            embeddings = [item["embedding"] for item in items]
        except (KeyError, TypeError) as exc:
            raise EmbeddingError(f"Unexpected Jina API response shape: {exc}") from exc

        self._validate_dimensions(embeddings)
        return embeddings

    def _validate_dimensions(self, embeddings: list[list[float]]) -> None:
        if not embeddings:
            return
        dims = {len(e) for e in embeddings}
        if len(dims) > 1:
            raise EmbeddingDimensionError(f"Inconsistent embedding dimensions in one batch: {dims}")

        actual_dim = next(iter(dims))
        if self.expected_dimension is not None and actual_dim != self.expected_dimension:
            raise EmbeddingDimensionError(
                f"Expected embedding dimension {self.expected_dimension}, got {actual_dim}"
            )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batch_embeddings = await self._embed_batch(batch, task="retrieval.passage")
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        embeddings = await self._embed_batch([query], task="retrieval.query")
        return embeddings[0]

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "JinaEmbedder":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
