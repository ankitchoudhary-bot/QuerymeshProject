"""Phase 25 unit tests: ndcg_at_k correctness, and the evaluation harness
(evaluate_retriever / compare_retrieval_methods) against fake search
callables — fast and isolated. The real benchmark against a genuine
retrieval stack lives in tests/evaluation/test_retrieval_benchmark.py.
"""

from __future__ import annotations

import math

import pytest

from querymesh.evaluation import (
    EvaluationCase,
    RetrievalComparisonReport,
    RetrievalMetrics,
    compare_retrieval_methods,
    evaluate_retriever,
    ndcg_at_k,
)
from querymesh.retrieval.models import RetrievedDocument


def _rd(chunk_id: str, rank: int) -> RetrievedDocument:
    return RetrievedDocument(document={"chunk_id": chunk_id}, score=1.0, rank=rank, retrieval_method="bm25")


# --- ndcg_at_k -----------------------------------------------------------


@pytest.mark.unit
def test_ndcg_perfect_ranking_is_one() -> None:
    # All 2 relevant docs at the top -> NDCG = 1.0
    retrieved = ["a", "b", "z"]
    relevant = {"a", "b"}
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)


@pytest.mark.unit
def test_ndcg_no_relevant_found_is_zero() -> None:
    assert ndcg_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


@pytest.mark.unit
def test_ndcg_worse_ranking_scores_lower_than_perfect() -> None:
    perfect = ndcg_at_k(["a", "b", "z"], {"a", "b"}, k=3)
    worse = ndcg_at_k(["z", "a", "b"], {"a", "b"}, k=3)
    assert worse < perfect
    assert worse > 0.0


@pytest.mark.unit
def test_ndcg_matches_manual_calculation() -> None:
    # relevant = {"a"} found at rank 2 (0-indexed position 1)
    # DCG = 1/log2(2+1) = 1/log2(3); IDCG = 1/log2(1+1) = 1/log2(2) = 1
    retrieved = ["z", "a", "y"]
    relevant = {"a"}
    expected = (1.0 / math.log2(3)) / 1.0
    assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(expected)


@pytest.mark.unit
def test_ndcg_empty_relevant_set_is_zero() -> None:
    assert ndcg_at_k(["a", "b"], set(), k=5) == 0.0


@pytest.mark.unit
def test_ndcg_zero_k_is_zero() -> None:
    assert ndcg_at_k(["a"], {"a"}, k=0) == 0.0


# --- evaluate_retriever ----------------------------------------------------


@pytest.mark.unit
async def test_evaluate_retriever_perfect_retrieval_scores_high() -> None:
    dataset = [
        EvaluationCase(query="q1", relevant_ids=["a"]),
        EvaluationCase(query="q2", relevant_ids=["b"]),
    ]

    async def perfect_search(query: str, top_k: int) -> list[RetrievedDocument]:
        chunk_id = "a" if query == "q1" else "b"
        return [_rd(chunk_id, 1)]

    metrics = await evaluate_retriever("test_method", perfect_search, dataset, top_k=10)

    assert metrics.method == "test_method"
    assert metrics.precision_at_1 == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_10 == pytest.approx(1.0)
    assert metrics.num_cases == 2


@pytest.mark.unit
async def test_evaluate_retriever_no_matches_scores_zero() -> None:
    dataset = [EvaluationCase(query="q1", relevant_ids=["a"])]

    async def empty_search(query: str, top_k: int) -> list[RetrievedDocument]:
        return [_rd("completely_unrelated", 1)]

    metrics = await evaluate_retriever("test_method", empty_search, dataset, top_k=10)

    assert metrics.precision_at_1 == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.mrr == 0.0
    assert metrics.ndcg_at_10 == 0.0


@pytest.mark.unit
async def test_evaluate_retriever_empty_dataset_returns_zeroed_metrics() -> None:
    async def search(query: str, top_k: int) -> list[RetrievedDocument]:
        return []

    metrics = await evaluate_retriever("test_method", search, [], top_k=10)

    assert metrics.num_cases == 0
    assert metrics.precision_at_1 == 0.0
    assert metrics.mrr == 0.0


@pytest.mark.unit
async def test_evaluate_retriever_averages_across_multiple_cases() -> None:
    dataset = [
        EvaluationCase(query="hit", relevant_ids=["a"]),
        EvaluationCase(query="miss", relevant_ids=["a"]),
    ]

    async def mixed_search(query: str, top_k: int) -> list[RetrievedDocument]:
        if query == "hit":
            return [_rd("a", 1)]
        return [_rd("nothing_relevant", 1)]

    metrics = await evaluate_retriever("test_method", mixed_search, dataset, top_k=10)

    assert metrics.precision_at_1 == pytest.approx(0.5)  # 1 hit, 1 miss
    assert metrics.mrr == pytest.approx(0.5)


# --- compare_retrieval_methods ----------------------------------------------


class _FakeBM25:
    async def search(self, query: str, top_k: int = 10) -> list[RetrievedDocument]:
        return [_rd("bm25_hit", 1)]


class _FakeVector:
    async def search(self, query_embedding: list[float], top_k: int = 10) -> list[RetrievedDocument]:
        return [_rd("vector_hit", 1)]


class _FakeHybrid:
    async def search(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        return [_rd("bm25_hit", 1), _rd("vector_hit", 2)]


class _FakeEmbedder:
    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


@pytest.mark.unit
async def test_compare_retrieval_methods_returns_all_three_reports() -> None:
    dataset = [EvaluationCase(query="q", relevant_ids=["bm25_hit", "vector_hit"])]

    report = await compare_retrieval_methods(
        bm25_retriever=_FakeBM25(),  # type: ignore[arg-type]
        vector_retriever=_FakeVector(),  # type: ignore[arg-type]
        hybrid_retriever=_FakeHybrid(),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        dataset=dataset,
    )

    assert isinstance(report, RetrievalComparisonReport)
    assert isinstance(report.bm25, RetrievalMetrics)
    assert report.bm25.method == "bm25"
    assert report.vector.method == "vector"
    assert report.hybrid.method == "hybrid_rrf"


@pytest.mark.unit
async def test_compare_retrieval_methods_hybrid_recalls_more_than_either_alone() -> None:
    """On this fake setup, hybrid returns both relevant docs while BM25 and
    Vector alone each only find one -- confirms the comparison harness
    reflects that real difference in the numbers.
    """
    dataset = [EvaluationCase(query="q", relevant_ids=["bm25_hit", "vector_hit"])]

    report = await compare_retrieval_methods(
        bm25_retriever=_FakeBM25(),  # type: ignore[arg-type]
        vector_retriever=_FakeVector(),  # type: ignore[arg-type]
        hybrid_retriever=_FakeHybrid(),  # type: ignore[arg-type]
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        dataset=dataset,
    )

    assert report.hybrid.recall_at_10 == pytest.approx(1.0)
    assert report.bm25.recall_at_10 == pytest.approx(0.5)
    assert report.vector.recall_at_10 == pytest.approx(0.5)
