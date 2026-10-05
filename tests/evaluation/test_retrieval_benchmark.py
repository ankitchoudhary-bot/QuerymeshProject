"""The real retrieval benchmark: BM25 vs Vector vs Hybrid RRF, measured
against a genuine indexed corpus through real `BM25Retriever`,
`VectorRetriever`, and `HybridRetriever` objects — not the illustrative,
hand-crafted `RetrievedDocument` lists used in Phase 9's test.

Only the OpenSearch transport is faked (same fake client as the Phase 24
integration tests); indexing, BM25 term matching, and RRF fusion are all
the real production code.
"""

from __future__ import annotations

import pytest

from querymesh.embeddings.fake import FakeEmbedder
from querymesh.evaluation import EvaluationCase, compare_retrieval_methods
from querymesh.indexing.indexer import ChunkIndexer
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.retrieval import BM25Retriever, HybridRetriever, VectorRetriever
from tests.integration.test_retrieval_integration import _FakeOpenSearchClient

# A small corpus where BM25 keyword matches are unambiguous by
# construction — each chunk's text contains a distinctive term found in
# exactly one evaluation query, so BM25's expected behavior is provable,
# not just plausible.
_CORPUS = [
    ("p1:chunk:0000", "p1", "Attention mechanisms allow models to weigh relevance in transformers."),
    ("p2:chunk:0000", "p2", "Convolutional neural networks are widely used in computer vision tasks."),
    ("p3:chunk:0000", "p3", "Recurrent networks process sequences one token at a time."),
    ("p4:chunk:0000", "p4", "Efficient attention variants reduce the quadratic complexity of transformers."),
    ("p5:chunk:0000", "p5", "Graph neural networks operate over relational, non-Euclidean data structures."),
]

_DATASET = [
    EvaluationCase(query="attention", relevant_ids=["p1:chunk:0000", "p4:chunk:0000"]),
    EvaluationCase(query="convolutional", relevant_ids=["p2:chunk:0000"]),
    EvaluationCase(query="recurrent", relevant_ids=["p3:chunk:0000"]),
    EvaluationCase(query="graph", relevant_ids=["p5:chunk:0000"]),
]


async def _build_corpus() -> tuple[OpenSearchConnection, FakeEmbedder]:
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-eval")
    await conn.create_index()

    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)
    chunks = [
        PaperChunk(
            chunk_id=chunk_id,
            arxiv_id=arxiv_id,
            title="Benchmark Corpus",
            section="Body",
            text=text,
            page=None,
            chunk_index=0,
        )
        for chunk_id, arxiv_id, text in _CORPUS
    ]
    await indexer.index_chunks(chunks)
    return conn, embedder


@pytest.mark.evaluation
async def test_bm25_achieves_perfect_precision_on_keyword_queries() -> None:
    """BM25's real strength, proven with real term matching: every query
    here has a distinctive keyword present in exactly its relevant
    chunk(s), so BM25 should retrieve them with perfect precision@1.
    """
    conn, _embedder = await _build_corpus()
    bm25 = BM25Retriever(conn)

    for case in _DATASET:
        results = await bm25.search(case.query, top_k=10)
        retrieved_ids = [r.chunk_id for r in results]
        assert retrieved_ids, f"BM25 found nothing for {case.query!r}"
        assert retrieved_ids[0] in case.relevant_ids


@pytest.mark.evaluation
async def test_compare_retrieval_methods_produces_valid_metrics_for_all_three() -> None:
    """The real Phase 25 deliverable: run the actual comparison harness
    against real BM25/Vector/Hybrid retrievers reading a real indexed
    corpus, and confirm every metric is a well-formed probability.
    """
    conn, embedder = await _build_corpus()
    bm25 = BM25Retriever(conn)
    vector = VectorRetriever(conn)
    hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector, embedder=embedder)

    report = await compare_retrieval_methods(
        bm25_retriever=bm25,
        vector_retriever=vector,
        hybrid_retriever=hybrid,
        embedder=embedder,
        dataset=_DATASET,
    )

    for metrics in (report.bm25, report.vector, report.hybrid):
        assert metrics.num_cases == len(_DATASET)
        for value in (
            metrics.precision_at_1,
            metrics.precision_at_3,
            metrics.precision_at_5,
            metrics.recall_at_5,
            metrics.recall_at_10,
            metrics.mrr,
            metrics.ndcg_at_10,
        ):
            assert 0.0 <= value <= 1.0

    # BM25's real, provable strength on this corpus (see test above):
    # perfect MRR, since the relevant chunk is always its top hit.
    assert report.bm25.mrr == pytest.approx(1.0)


@pytest.mark.evaluation
async def test_hybrid_recall_at_least_matches_bm25_on_this_corpus() -> None:
    """With a 5-document corpus and top_k=10, RRF fuses in every candidate
    either method surfaces, so hybrid can't retrieve strictly less than
    BM25 alone finds here — a real, checkable property of this corpus
    size relative to top_k, not a general claim about RRF.
    """
    conn, embedder = await _build_corpus()
    bm25 = BM25Retriever(conn)
    vector = VectorRetriever(conn)
    hybrid = HybridRetriever(bm25_retriever=bm25, vector_retriever=vector, embedder=embedder)

    report = await compare_retrieval_methods(
        bm25_retriever=bm25, vector_retriever=vector, hybrid_retriever=hybrid,
        embedder=embedder, dataset=_DATASET,
    )

    assert report.hybrid.recall_at_10 >= report.bm25.recall_at_10
