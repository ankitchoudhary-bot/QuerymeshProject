"""Unified retrieval service.

This is the *only* retrieval-layer object agents (Phase 11+) should ever
depend on. It hides embedding, BM25, vector search, RRF, and metadata
filtering — callers pass a query (and optional filters) and get back
ranked `RetrievedDocument`s, with no knowledge of how retrieval works
underneath.

When a `tracer` is supplied, `retrieve()` opens a "retrieval" span; the
"bm25"/"vector"/"rrf" spans `HybridRetriever` (Phase 9/18) opens nest
inside it automatically via the underlying span-context mechanism, giving
the exact hierarchy from the observability spec.
"""

from __future__ import annotations

from querymesh.errors import QueryMeshError, RetrievalError
from querymesh.observability.logging import get_logger
from querymesh.observability.tracing import NoOpTracer, Tracer
from querymesh.retrieval.hybrid import HybridRetriever
from querymesh.retrieval.models import RetrievedDocument
from querymesh.security.limits import validate_query, validate_top_k

logger = get_logger(__name__)


class RetrievalService:
    """Facade over hybrid (BM25 + vector + RRF) retrieval."""

    def __init__(self, hybrid_retriever: HybridRetriever, tracer: Tracer | None = None) -> None:
        self._hybrid_retriever = hybrid_retriever
        self.tracer = tracer or NoOpTracer()

    async def retrieve(self, query: str, top_k: int = 10, filters: dict[str, str | list[str]] | None = None,) -> list[RetrievedDocument]:
        """Retrieve the top_k most relevant chunks for `query`.

        `filters` is an optional metadata filter, e.g.
        `{"categories": ["cs.CL", "cs.LG"]}` or `{"arxiv_id": "2608.13558v1"}`.
        """
        query = validate_query(query)
        top_k = validate_top_k(top_k)

        with self.tracer.span("retrieval", query=query, top_k=top_k):
            try:
                return await self._hybrid_retriever.search(query, top_k=top_k, filters=filters)
            except QueryMeshError:
                raise  # already a well-typed error (e.g. from OpenSearch layer) — don't rewrap
            except Exception as exc:  # noqa: BLE001 - surface as a clear retrieval-layer error
                logger.error("Retrieval failed for query %r: %s", query, exc)
                raise RetrievalError(f"Retrieval failed: {exc}") from exc
