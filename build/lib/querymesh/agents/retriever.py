"""Retrieval agent.

Responsibilities: execute retrieval (possibly across several query
variations from the Researcher agent) and return deduplicated candidate
chunks.
"""

from __future__ import annotations

from querymesh.errors import AgentError
from querymesh.observability.logging import get_logger
from querymesh.retrieval.models import RetrievedDocument
from querymesh.retrieval.service import RetrievalService

logger = get_logger(__name__)


class RetrieverAgent:
    """Executes retrieval for one or more search queries via RetrievalService."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        if retrieval_service is None:
            raise AgentError("RetrieverAgent requires a non-None retrieval_service")
        self.retrieval_service = retrieval_service

    async def retrieve(
        self,
        queries: list[str],
        top_k: int = 10,
        filters: dict[str, str | list[str]] | None = None,
    ) -> list[RetrievedDocument]:
        """Run retrieval for each query, merging and deduplicating by chunk_id.

        When a chunk_id appears under multiple queries, the highest-scoring
        occurrence is kept.
        """
        best_by_id: dict[str, RetrievedDocument] = {}
        unidentified: list[RetrievedDocument] = []

        for query in queries:
            results = await self.retrieval_service.retrieve(query, top_k=top_k, filters=filters)
            for result in results:
                chunk_id = result.chunk_id
                if chunk_id is None:
                    unidentified.append(result)
                    continue
                existing = best_by_id.get(chunk_id)
                if existing is None or result.score > existing.score:
                    best_by_id[chunk_id] = result

        merged = list(best_by_id.values()) + unidentified
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged
