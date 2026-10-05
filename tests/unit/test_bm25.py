"""Phase 6 tests: BM25Retriever query construction/parsing, and a small
retrieval-evaluation dataset exercising Precision@K, Recall@K, MRR.
"""

from __future__ import annotations

import pytest

from querymesh.evaluation import (
    EvaluationCase,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.retrieval import BM25Retriever, RetrievedDocument


class _FakeConnection:
    """Records the search body it receives and returns a canned response."""

    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.last_body: dict | None = None

    async def search(self, body: dict) -> dict:
        self.last_body = body
        size = body.get("size", len(self._hits))
        return {"hits": {"hits": self._hits[:size]}}


def _hit(chunk_id: str, score: float, text: str = "some text") -> dict:
    return {"_id": chunk_id, "_score": score, "_source": {"chunk_id": chunk_id, "text": text}}


@pytest.mark.unit
async def test_bm25_search_builds_multi_match_query() -> None:
    conn = _FakeConnection(hits=[])
    retriever = BM25Retriever(connection=conn)  # type: ignore[arg-type]

    await retriever.search("transformer attention", top_k=5)

    assert conn.last_body is not None
    assert conn.last_body["size"] == 5
    query = conn.last_body["query"]["multi_match"]
    assert query["query"] == "transformer attention"
    assert "title^2" in query["fields"]
    assert "text" in query["fields"]


@pytest.mark.unit
async def test_bm25_search_parses_hits_into_retrieved_documents() -> None:
    hits = [
        _hit("a:chunk:0000", score=12.4),
        _hit("a:chunk:0001", score=9.1),
    ]
    conn = _FakeConnection(hits=hits)
    retriever = BM25Retriever(connection=conn)  # type: ignore[arg-type]

    results = await retriever.search("hybrid retrieval", top_k=10)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedDocument) for r in results)
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[0].score == 12.4
    assert results[0].retrieval_method == "bm25"
    assert results[0].chunk_id == "a:chunk:0000"


@pytest.mark.unit
async def test_bm25_search_respects_top_k() -> None:
    hits = [_hit(f"a:chunk:{i:04d}", score=float(10 - i)) for i in range(5)]
    conn = _FakeConnection(hits=hits)
    retriever = BM25Retriever(connection=conn)  # type: ignore[arg-type]

    results = await retriever.search("query", top_k=2)
    assert len(results) == 2


@pytest.mark.unit
async def test_bm25_search_handles_no_hits() -> None:
    conn = _FakeConnection(hits=[])
    retriever = BM25Retriever(connection=conn)  # type: ignore[arg-type]

    results = await retriever.search("nothing matches this", top_k=10)
    assert results == []


# --- Metrics -----------------------------------------------------------


@pytest.mark.unit
def test_precision_at_k_basic() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, k=4) == 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert precision_at_k(retrieved, relevant, k=1) == 1.0


@pytest.mark.unit
def test_recall_at_k_basic() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c", "z"}
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)


@pytest.mark.unit
def test_mrr_basic() -> None:
    assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0


@pytest.mark.unit
def test_metrics_handle_empty_inputs() -> None:
    assert precision_at_k([], {"a"}, k=5) == 0.0
    assert recall_at_k(["a"], set(), k=5) == 0.0
    assert mean_reciprocal_rank([], {"a"}) == 0.0


@pytest.mark.unit
async def test_small_evaluation_dataset_end_to_end() -> None:
    """A tiny 2-query evaluation, as the phase spec asks for, using canned
    BM25 results (no live cluster) to compute Precision@K/Recall@K/MRR.
    """
    dataset = [
        EvaluationCase(query="transformer attention mechanism", relevant_ids=["p1:chunk:0000"]),
        EvaluationCase(query="hybrid retrieval fusion", relevant_ids=["p2:chunk:0001", "p2:chunk:0002"]),
    ]

    canned_hits = {
        "transformer attention mechanism": [
            _hit("p1:chunk:0000", 15.0),
            _hit("p3:chunk:0000", 8.0),
        ],
        "hybrid retrieval fusion": [
            _hit("p2:chunk:0002", 11.0),
            _hit("p9:chunk:0000", 7.0),
            _hit("p2:chunk:0001", 5.0),
        ],
    }

    precisions, recalls, rr_scores = [], [], []
    for case in dataset:
        conn = _FakeConnection(hits=canned_hits[case.query])
        retriever = BM25Retriever(connection=conn)  # type: ignore[arg-type]
        results = await retriever.search(case.query, top_k=5)
        retrieved_ids = [r.chunk_id for r in results]
        relevant = set(case.relevant_ids)

        precisions.append(precision_at_k(retrieved_ids, relevant, k=5))
        recalls.append(recall_at_k(retrieved_ids, relevant, k=5))
        rr_scores.append(mean_reciprocal_rank(retrieved_ids, relevant))

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mrr = sum(rr_scores) / len(rr_scores)

    assert mean_precision > 0
    assert mean_recall > 0
    assert mrr > 0
    # Case 1: 1 of 2 retrieved is relevant -> precision 0.5, recall 1.0, RR 1.0
    # Case 2: 2 of 3 retrieved are relevant -> precision ~0.667, recall 1.0, RR 1.0
    assert recalls == [1.0, 1.0]
    assert rr_scores == [1.0, 1.0]
