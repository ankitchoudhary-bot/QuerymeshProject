"""Phase 10 tests: metadata filter clause building, filter pass-through in
BM25/vector/hybrid retrievers, and the RetrievalService facade.
"""

from __future__ import annotations

import pytest

from querymesh.embeddings.fake import FakeEmbedder
from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever
from querymesh.retrieval.filters import build_filter_clauses
from querymesh.retrieval.models import RetrievedDocument


def _rd(chunk_id: str, rank: int, method: str) -> RetrievedDocument:
    return RetrievedDocument(
        document={"chunk_id": chunk_id, "text": "x"}, score=0.0, rank=rank, retrieval_method=method
    )


class _RecordingConnection:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.last_body: dict | None = None

    async def search(self, body: dict) -> dict:
        self.last_body = body
        return {"hits": {"hits": self._hits}}


# --- filter clause builder ------------------------------------------------


@pytest.mark.unit
def test_build_filter_clauses_single_value() -> None:
    clauses = build_filter_clauses({"arxiv_id": "2608.13558v1"})
    assert clauses == [{"term": {"arxiv_id": "2608.13558v1"}}]


@pytest.mark.unit
def test_build_filter_clauses_list_value() -> None:
    clauses = build_filter_clauses({"categories": ["cs.CL", "cs.LG"]})
    assert clauses == [{"terms": {"categories": ["cs.CL", "cs.LG"]}}]


@pytest.mark.unit
def test_build_filter_clauses_multiple_fields() -> None:
    clauses = build_filter_clauses({"arxiv_id": "a1", "categories": ["cs.CL"]})
    assert len(clauses) == 2


# --- BM25 / vector filter pass-through -------------------------------------


@pytest.mark.unit
async def test_bm25_search_without_filters_keeps_plain_query_shape() -> None:
    conn = _RecordingConnection(hits=[])
    retriever = BM25Retriever(connection=conn)  # type: ignore[arg-type]

    await retriever.search("attention", top_k=5)

    assert "multi_match" in conn.last_body["query"]
    assert "bool" not in conn.last_body["query"]


@pytest.mark.unit
async def test_bm25_search_with_filters_wraps_in_bool_query() -> None:
    conn = _RecordingConnection(hits=[])
    retriever = BM25Retriever(connection=conn)  # type: ignore[arg-type]

    await retriever.search("attention", top_k=5, filters={"arxiv_id": "2608.13558v1"})

    bool_query = conn.last_body["query"]["bool"]
    assert bool_query["must"][0]["multi_match"]["query"] == "attention"
    assert bool_query["filter"] == [{"term": {"arxiv_id": "2608.13558v1"}}]


@pytest.mark.unit
async def test_vector_search_without_filters_keeps_plain_knn_shape() -> None:
    conn = _RecordingConnection(hits=[])
    retriever = VectorRetriever(connection=conn)  # type: ignore[arg-type]

    await retriever.search([0.1, 0.2], top_k=5)

    assert "filter" not in conn.last_body["query"]["knn"]["embedding"]


@pytest.mark.unit
async def test_vector_search_with_filters_adds_filter_clause() -> None:
    conn = _RecordingConnection(hits=[])
    retriever = VectorRetriever(connection=conn)  # type: ignore[arg-type]

    await retriever.search([0.1, 0.2], top_k=5, filters={"categories": ["cs.CL"]})

    knn_filter = conn.last_body["query"]["knn"]["embedding"]["filter"]
    assert knn_filter == {"bool": {"filter": [{"terms": {"categories": ["cs.CL"]}}]}}


# --- RetrievalService -------------------------------------------------------


class _FakeHybrid:
    def __init__(self, results: list[RetrievedDocument]) -> None:
        self._results = results
        self.called_with: dict | None = None

    async def search(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedDocument]:
        self.called_with = {"query": query, "top_k": top_k, "filters": filters}
        return self._results[:top_k]


@pytest.mark.unit
async def test_retrieval_service_delegates_to_hybrid_retriever() -> None:
    results = [_rd("a", 1, "hybrid_rrf"), _rd("b", 2, "hybrid_rrf")]
    fake_hybrid = _FakeHybrid(results)
    service = RetrievalService(hybrid_retriever=fake_hybrid)  # type: ignore[arg-type]

    out = await service.retrieve("transformer attention", top_k=2, filters={"arxiv_id": "a1"})

    assert out == results
    assert fake_hybrid.called_with == {
        "query": "transformer attention",
        "top_k": 2,
        "filters": {"arxiv_id": "a1"},
    }


@pytest.mark.unit
async def test_retrieval_service_default_filters_is_none() -> None:
    fake_hybrid = _FakeHybrid([])
    service = RetrievalService(hybrid_retriever=fake_hybrid)  # type: ignore[arg-type]

    await service.retrieve("query")

    assert fake_hybrid.called_with["filters"] is None


@pytest.mark.unit
async def test_retrieval_service_end_to_end_with_real_hybrid_retriever() -> None:
    """RetrievalService wired to a real HybridRetriever (with fake BM25/
    vector connections) — proves the full facade hides embedding + BM25 +
    vector + RRF behind one call.
    """
    bm25_conn = _RecordingConnection(hits=[{"_id": "a", "_score": 5.0, "_source": {"chunk_id": "a"}}])
    vector_conn = _RecordingConnection(hits=[{"_id": "a", "_score": 0.9, "_source": {"chunk_id": "a"}}])

    hybrid = HybridRetriever(
        bm25_retriever=BM25Retriever(connection=bm25_conn),  # type: ignore[arg-type]
        vector_retriever=VectorRetriever(connection=vector_conn),  # type: ignore[arg-type]
        embedder=FakeEmbedder(dimension=8),
    )
    service = RetrievalService(hybrid_retriever=hybrid)

    results = await service.retrieve("hybrid retrieval fusion", top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "a"
    assert results[0].retrieval_method == "hybrid_rrf"
