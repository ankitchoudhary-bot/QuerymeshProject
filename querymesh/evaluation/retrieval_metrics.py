"""Generic retrieval evaluation metrics.

Deliberately independent of BM25/vector/hybrid specifics — takes a ranked
list of retrieved IDs and a set of relevant IDs. Phases 8/9 (vector,
hybrid RRF) and Phase 25 (the full evaluation harness) reuse this as-is.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    """One query + its known-relevant document/chunk IDs."""

    query: str
    relevant_ids: list[str] = Field(default_factory=list)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain, binary relevance.

    DCG discounts a hit by 1/log2(rank+1); IDCG is the DCG of the best
    possible ordering (all relevant docs first, up to k). NDCG = DCG/IDCG,
    0.0 when there's nothing relevant to find.
    """
    if k <= 0 or not relevant_ids:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)  # i is 0-indexed; rank = i+1, so denominator is log2(rank+1)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0
