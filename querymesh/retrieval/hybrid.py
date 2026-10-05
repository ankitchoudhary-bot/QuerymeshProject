"""Hybrid retrieval: BM25 + vector search, fused with RRF.

Query
 ├── BM25
 │
 └── Vector Search
 ↓
 RRF
 ↓
 Final ranked documents

When a `tracer` is supplied, each stage gets its own span ("bm25",
"vector", "rrf") — the children of the "retrieval" span RetrievalService
(Phase 10/18) opens, matching the observability hierarchy from the spec.
"""

from __future__ import annotations

from querymesh.embeddings.base import Embedder
from querymesh.observability.tracing import NoOpTracer, Tracer
from querymesh.retrieval.bm25 import BM25Retriever
from querymesh.retrieval.models import RetrievedDocument
from querymesh.retrieval.rrf import reciprocal_rank_fusion
from querymesh.retrieval.vector import VectorRetriever

_DEFAULT_RRF_K = 60


class HybridRetriever:
    """Combines `BM25Retriever` and `VectorRetriever` results via RRF."""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        embedder: Embedder,
        rrf_k: int = _DEFAULT_RRF_K,
        tracer: Tracer | None = None,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.tracer = tracer or NoOpTracer()

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str | list[str]] | None = None,
    ) -> list[RetrievedDocument]:
        with self.tracer.span("bm25", query=query, top_k=top_k):
            bm25_results = await self.bm25_retriever.search(query, top_k=top_k, filters=filters)

        with self.tracer.span("vector", top_k=top_k):
            query_embedding = await self.embedder.embed_query(query)
            vector_results = await self.vector_retriever.search(
                query_embedding, top_k=top_k, filters=filters
            )

        with self.tracer.span("rrf", k=self.rrf_k):
            fused = reciprocal_rank_fusion([bm25_results, vector_results], k=self.rrf_k)

        return fused[:top_k]
