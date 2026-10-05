"""BM25 (lexical) retrieval.

QueryMesh rule for this phase: "Make BM25 work. Add a small test. Document
known limitations. Move to vector retrieval." — no parameter tuning here.
"""

from __future__ import annotations

from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.retrieval.filters import build_filter_clauses
from querymesh.retrieval.models import RetrievedDocument


class BM25Retriever:
    """Lexical retrieval via OpenSearch's default BM25 similarity."""

    def __init__(self, connection: OpenSearchConnection) -> None:
        self.connection = connection

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, str | list[str]] | None = None,
    ) -> list[RetrievedDocument]:
        multi_match = {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "text", "section", "abstract"],
            }
        }

        if filters:
            search_query = {"bool": {"must": [multi_match], "filter": build_filter_clauses(filters)}}
        else:
            search_query = multi_match

        body = {"size": top_k, "query": search_query}
        response = await self.connection.search(body)
        hits = response.get("hits", {}).get("hits", [])

        return [
            RetrievedDocument(
                document=hit.get("_source", {}),
                score=float(hit.get("_score") or 0.0),
                rank=rank,
                retrieval_method="bm25",
            )
            for rank, hit in enumerate(hits, start=1)
        ]
