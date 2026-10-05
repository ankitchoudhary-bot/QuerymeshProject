"""Vector (k-NN) retrieval.

Returns the same `RetrievedDocument` shape `BM25Retriever` does — Phase 9
(RRF) fuses the two without caring which one produced a given result.
"""

from __future__ import annotations

from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.retrieval.filters import build_filter_clauses
from querymesh.retrieval.models import RetrievedDocument


class VectorRetriever:
    """Semantic retrieval via OpenSearch's k-NN vector search."""

    def __init__(self, connection: OpenSearchConnection, vector_field: str = "embedding") -> None:
        self.connection = connection
        self.vector_field = vector_field

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, str | list[str]] | None = None,
    ) -> list[RetrievedDocument]:
        knn_clause: dict = {
            self.vector_field: {
                "vector": query_embedding,
                "k": top_k,
            }
        }
        if filters:
            knn_clause[self.vector_field]["filter"] = {"bool": {"filter": build_filter_clauses(filters)}}

        body = {"size": top_k, "query": {"knn": knn_clause}}
        response = await self.connection.search(body)
        hits = response.get("hits", {}).get("hits", [])

        return [
            RetrievedDocument(
                document=hit.get("_source", {}),
                score=float(hit.get("_score") or 0.0),
                rank=rank,
                retrieval_method="vector",
            )
            for rank, hit in enumerate(hits, start=1)
        ]
