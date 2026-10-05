"""Phase 9 tests: reciprocal_rank_fusion correctness, HybridRetriever
composition, and a small comparison of BM25 vs vector vs hybrid on the
Phase 6 evaluation dataset (not a benchmark — just confirms hybrid can
recover documents either method alone misses).
"""

from __future__ import annotations

import pytest

from querymesh.embeddings.fake import FakeEmbedder
from querymesh.evaluation import mean_reciprocal_rank, precision_at_k, recall_at_k
from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievedDocument, VectorRetriever
from querymesh.retrieval.rrf import reciprocal_rank_fusion


def _rd(chunk_id: str, rank: int, method: str) -> RetrievedDocument:
    return RetrievedDocument(
        document={"chunk_id": chunk_id, "text": "x"}, score=0.0, rank=rank, retrieval_method=method
    )


# --- reciprocal_rank_fusion --------------------------------------------


@pytest.mark.unit
def test_rrf_matches_formula_for_single_list() -> None:
    results = [_rd("a", 1, "bm25"), _rd("b", 2, "bm25")]
    fused = reciprocal_rank_fusion([results], k=60)

    assert fused[0].chunk_id == "a"
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].chunk_id == "b"
    assert fused[1].score == pytest.approx(1 / 62)


@pytest.mark.unit
def test_rrf_combines_scores_for_docs_in_both_lists() -> None:
    bm25 = [_rd("a", 1, "bm25"), _rd("b", 2, "bm25")]
    vector = [_rd("b", 1, "vector"), _rd("a", 2, "vector")]

    fused = reciprocal_rank_fusion([bm25, vector], k=60)

    # "a": 1/(60+1) + 1/(60+2) ; "b": 1/(60+2) + 1/(60+1) -> tied
    expected = 1 / 61 + 1 / 62
    assert fused[0].score == pytest.approx(expected)
    assert fused[1].score == pytest.approx(expected)
    assert {fused[0].chunk_id, fused[1].chunk_id} == {"a", "b"}


@pytest.mark.unit
def test_rrf_promotes_doc_appearing_in_both_lists_over_single_list_doc() -> None:
    bm25 = [_rd("only_bm25", 1, "bm25"), _rd("shared", 2, "bm25")]
    vector = [_rd("shared", 1, "vector"), _rd("only_vector", 2, "vector")]

    fused = reciprocal_rank_fusion([bm25, vector], k=60)

    assert fused[0].chunk_id == "shared"
    assert fused[0].retrieval_method == "hybrid_rrf"


@pytest.mark.unit
def test_rrf_drops_documents_without_chunk_id() -> None:
    no_id = RetrievedDocument(document={"text": "no id"}, score=0.0, rank=1, retrieval_method="bm25")
    with_id = _rd("has_id", 2, "bm25")

    fused = reciprocal_rank_fusion([[no_id, with_id]], k=60)

    assert len(fused) == 1
    assert fused[0].chunk_id == "has_id"


@pytest.mark.unit
def test_rrf_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.mark.unit
def test_rrf_ranks_are_sequential_and_one_indexed() -> None:
    bm25 = [_rd("a", 1, "bm25"), _rd("b", 2, "bm25"), _rd("c", 3, "bm25")]
    fused = reciprocal_rank_fusion([bm25], k=60)
    assert [r.rank for r in fused] == [1, 2, 3]


# --- HybridRetriever ------------------------------------------------------


class _FakeBM25:
    def __init__(self, results: list[RetrievedDocument]) -> None:
        self._results = results
        self.called_with: str | None = None
        self.called_with_filters: dict | None = None

    async def search(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedDocument]:
        self.called_with = query
        self.called_with_filters = filters
        return self._results[:top_k]


class _FakeVector:
    def __init__(self, results: list[RetrievedDocument]) -> None:
        self._results = results
        self.called_with_embedding: list[float] | None = None
        self.called_with_filters: dict | None = None

    async def search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[RetrievedDocument]:
        self.called_with_embedding = query_embedding
        self.called_with_filters = filters
        return self._results[:top_k]


@pytest.mark.unit
async def test_hybrid_retriever_fuses_bm25_and_vector_results() -> None:
    bm25_results = [_rd("shared", 1, "bm25"), _rd("bm25_only", 2, "bm25")]
    vector_results = [_rd("shared", 1, "vector"), _rd("vector_only", 2, "vector")]

    bm25 = _FakeBM25(bm25_results)
    vector = _FakeVector(vector_results)
    embedder = FakeEmbedder(dimension=8)

    hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector, embedder=embedder)  # type: ignore[arg-type]
    results = await hybrid.search("transformer attention", top_k=10)

    ids = [r.chunk_id for r in results]
    assert "shared" in ids
    assert "bm25_only" in ids
    assert "vector_only" in ids
    assert ids[0] == "shared"  # appears in both, should rank first
    assert all(r.retrieval_method == "hybrid_rrf" for r in results)
    assert bm25.called_with == "transformer attention"


@pytest.mark.unit
async def test_hybrid_retriever_respects_top_k() -> None:
    bm25_results = [_rd(f"b{i}", i, "bm25") for i in range(1, 6)]
    vector_results = [_rd(f"v{i}", i, "vector") for i in range(1, 6)]

    hybrid = HybridRetriever(
        bm25_retriever=_FakeBM25(bm25_results),  # type: ignore[arg-type]
        vector_retriever=_FakeVector(vector_results),  # type: ignore[arg-type]
        embedder=FakeEmbedder(dimension=8),
    )
    results = await hybrid.search("query", top_k=3)
    assert len(results) == 3


# --- Comparison: BM25 vs vector vs hybrid on a tiny eval dataset ------------


@pytest.mark.unit
async def test_hybrid_recall_covers_docs_either_method_alone_misses() -> None:
    """Hybrid should recover the union of relevant docs found by BM25 and
    vector search alone — the whole point of fusing them. Not a tuning
    benchmark, just confirms the expected qualitative behavior.
    """
    relevant = {"p1:chunk:0000", "p1:chunk:0001"}

    # BM25 finds one relevant doc, misses the other.
    bm25_results = [_rd("p1:chunk:0000", 1, "bm25"), _rd("noise:chunk:0000", 2, "bm25")]
    # Vector finds the other relevant doc, misses the first.
    vector_results = [_rd("p1:chunk:0001", 1, "vector"), _rd("noise:chunk:0001", 2, "vector")]

    bm25_ids = [r.chunk_id for r in bm25_results]
    vector_ids = [r.chunk_id for r in vector_results]

    hybrid = HybridRetriever(
        bm25_retriever=_FakeBM25(bm25_results),  # type: ignore[arg-type]
        vector_retriever=_FakeVector(vector_results),  # type: ignore[arg-type]
        embedder=FakeEmbedder(dimension=8),
    )
    hybrid_results = await hybrid.search("query", top_k=10)
    hybrid_ids = [r.chunk_id for r in hybrid_results]

    bm25_recall = recall_at_k(bm25_ids, relevant, k=10)
    vector_recall = recall_at_k(vector_ids, relevant, k=10)
    hybrid_recall = recall_at_k(hybrid_ids, relevant, k=10)

    assert bm25_recall == pytest.approx(0.5)
    assert vector_recall == pytest.approx(0.5)
    assert hybrid_recall == pytest.approx(1.0)  # hybrid recovers both

    hybrid_mrr = mean_reciprocal_rank(hybrid_ids, relevant)
    assert hybrid_mrr == 1.0
