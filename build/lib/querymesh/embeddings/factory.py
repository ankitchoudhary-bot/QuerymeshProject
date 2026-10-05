"""Embedder factory — resolves a configured `Embedder` from `Settings`.

Mirrors `querymesh.llm.factory.get_llm`: falls back to `FakeEmbedder` when
no Jina API key is configured, so the rest of the pipeline (chunking ->
indexing -> retrieval) can be built and tested before real credentials
exist.
"""

from __future__ import annotations

from querymesh.config import Settings, get_settings
from querymesh.embeddings.base import Embedder, EmbeddingError
from querymesh.embeddings.fake import FakeEmbedder
from querymesh.embeddings.jina import JinaEmbedder

_DEFAULT_FAKE_DIMENSION = 64


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()

    if not settings.jina_api_key:
        return FakeEmbedder(dimension=_DEFAULT_FAKE_DIMENSION)

    if not settings.jina_embedding_model:
        raise EmbeddingError("JINA_EMBEDDING_MODEL must be set when JINA_API_KEY is configured")

    return JinaEmbedder(api_key=settings.jina_api_key, model=settings.jina_embedding_model)
