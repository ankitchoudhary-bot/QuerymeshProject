"""Retrieval evaluation harness.

Runs a benchmark dataset (list of `EvaluationCase`) against any retriever
exposing an async `search(...) -> list[RetrievedDocument]`-shaped method,
and aggregates Precision@1/3/5, Recall@5/10, MRR, and NDCG@10 across the
dataset. `compare_retrieval_methods` runs the same dataset against BM25,
Vector, and Hybrid retrievers so the three can be compared on real
measurements — not the illustrative canned-result comparison from Phase 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from querymesh.embeddings.base import Embedder
from querymesh.evaluation.retrieval_metrics import (
    EvaluationCase,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from querymesh.retrieval.bm25 import BM25Retriever
from querymesh.retrieval.hybrid import HybridRetriever
from querymesh.retrieval.models import RetrievedDocument
from querymesh.retrieval.vector import VectorRetriever

_DEFAULT_TOP_K = 10


@dataclass
class RetrievalMetrics:
    """Mean metrics across an evaluation dataset for one retrieval method."""

    method: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    num_cases: int


@dataclass
class RetrievalComparisonReport:
    bm25: RetrievalMetrics
    vector: RetrievalMetrics
    hybrid: RetrievalMetrics


async def evaluate_retriever(
    method_name: str,
    search: Callable[[str, int], Awaitable[list[RetrievedDocument]]],
    dataset: list[EvaluationCase],
    top_k: int = _DEFAULT_TOP_K,
) -> RetrievalMetrics:
    """Run `search(query, top_k)` over every case in `dataset` and average
    each metric across cases. A case with no relevant_ids contributes 0.0
    for every metric (can't score recall/NDCG without a relevance set) —
    keep the benchmark dataset free of such cases for meaningful numbers.
    """
    if not dataset:
        return RetrievalMetrics(method_name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    p1s, p3s, p5s, r5s, r10s, rrs, ndcgs = [], [], [], [], [], [], []

    for case in dataset:
        results = await search(case.query, top_k)
        retrieved_ids = [r.chunk_id for r in results if r.chunk_id]
        relevant = set(case.relevant_ids)

        p1s.append(precision_at_k(retrieved_ids, relevant, k=1))
        p3s.append(precision_at_k(retrieved_ids, relevant, k=3))
        p5s.append(precision_at_k(retrieved_ids, relevant, k=5))
        r5s.append(recall_at_k(retrieved_ids, relevant, k=5))
        r10s.append(recall_at_k(retrieved_ids, relevant, k=10))
        rrs.append(mean_reciprocal_rank(retrieved_ids, relevant))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, k=10))

    n = len(dataset)
    return RetrievalMetrics(
        method=method_name,
        precision_at_1=sum(p1s) / n,
        precision_at_3=sum(p3s) / n,
        precision_at_5=sum(p5s) / n,
        recall_at_5=sum(r5s) / n,
        recall_at_10=sum(r10s) / n,
        mrr=sum(rrs) / n,
        ndcg_at_10=sum(ndcgs) / n,
        num_cases=n,
    )


async def compare_retrieval_methods(
    bm25_retriever: BM25Retriever,
    vector_retriever: VectorRetriever,
    hybrid_retriever: HybridRetriever,
    embedder: Embedder,
    dataset: list[EvaluationCase],
    top_k: int = _DEFAULT_TOP_K,
) -> RetrievalComparisonReport:
    """Evaluate BM25, Vector, and Hybrid retrieval on the same dataset."""

    async def _bm25_search(query: str, k: int) -> list[RetrievedDocument]:
        return await bm25_retriever.search(query, top_k=k)

    async def _vector_search(query: str, k: int) -> list[RetrievedDocument]:
        query_embedding = await embedder.embed_query(query)
        return await vector_retriever.search(query_embedding, top_k=k)

    async def _hybrid_search(query: str, k: int) -> list[RetrievedDocument]:
        return await hybrid_retriever.search(query, top_k=k)

    bm25_metrics = await evaluate_retriever("bm25", _bm25_search, dataset, top_k)
    vector_metrics = await evaluate_retriever("vector", _vector_search, dataset, top_k)
    hybrid_metrics = await evaluate_retriever("hybrid_rrf", _hybrid_search, dataset, top_k)

    return RetrievalComparisonReport(bm25=bm25_metrics, vector=vector_metrics, hybrid=hybrid_metrics)
