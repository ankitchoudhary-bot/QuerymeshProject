"""Integration test: BM25Retriever + VectorRetriever + HybridRetriever +
RetrievalService + ChunkIndexer, all real objects wired together, with the
external boundary mocked at the lowest possible point — a fake OpenSearch
client implementing the same protocol the real `opensearchpy.AsyncOpenSearch`
does. Unlike the unit tests (which test each retriever in isolation with a
canned search response), this exercises the real composition: indexing
writes real documents into the fake store, and retrieval reads them back
through the full BM25 -> vector -> RRF -> RetrievalService chain.
"""

from __future__ import annotations

import pytest

from querymesh.embeddings.fake import FakeEmbedder
from querymesh.indexing.indexer import ChunkIndexer
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever


class _FakeIndicesClient:
    def __init__(self) -> None:
        self._indices: dict[str, dict] = {}

    async def exists(self, index: str) -> bool:
        return index in self._indices

    async def create(self, index: str, body: dict) -> dict:
        self._indices[index] = body
        return {"acknowledged": True}

    async def delete(self, index: str) -> dict:
        self._indices.pop(index, None)
        return {"acknowledged": True}


class _FakeOpenSearchClient:
    """In-memory client supporting real bulk-index + naive substring/kNN
    search, so retrieval integration tests exercise genuine round-trips
    (write via ChunkIndexer, read via BM25Retriever/VectorRetriever)
    instead of canned responses.
    """

    def __init__(self) -> None:
        self.indices = _FakeIndicesClient()
        self._docs: dict[str, dict[str, dict]] = {}

    async def ping(self) -> bool:
        return True

    async def bulk(self, *, body: list) -> dict:
        items = []
        for i in range(0, len(body), 2):
            action = body[i]["index"]
            source = body[i + 1]
            self._docs.setdefault(action["_index"], {})[action["_id"]] = source
            items.append({"index": {"_id": action["_id"], "status": 201}})
        return {"errors": False, "items": items}

    async def search(self, *, index: str, body: dict) -> dict:
        docs = self._docs.get(index, {})
        query = body.get("query", {})

        multi_match = _extract_multi_match(query)
        if multi_match is not None:
            terms = multi_match["query"].lower().split()
            hits = [
                {"_id": doc_id, "_score": 1.0, "_source": src}
                for doc_id, src in docs.items()
                if any(term in src.get("text", "").lower() for term in terms)
            ]
        elif "knn" in query:
            # Naive kNN stand-in: return all docs with a fake descending score
            # (real similarity math isn't the point of this integration test;
            # exercising the real request/response plumbing is).
            hits = [
                {"_id": doc_id, "_score": 1.0 - i * 0.01, "_source": src}
                for i, (doc_id, src) in enumerate(docs.items())
            ]
        else:
            hits = [{"_id": doc_id, "_score": 1.0, "_source": src} for doc_id, src in docs.items()]

        size = body.get("size", len(hits))
        return {"hits": {"hits": hits[:size]}}

    async def close(self) -> None:
        pass


def _extract_multi_match(query: dict) -> dict | None:
    if "multi_match" in query:
        return query["multi_match"]
    for clause in query.get("bool", {}).get("must", []):
        if "multi_match" in clause:
            return clause["multi_match"]
    return None


def _chunk(index: int, text: str, arxiv_id: str = "p1") -> PaperChunk:
    return PaperChunk(
        chunk_id=f"{arxiv_id}:chunk:{index:04d}",
        arxiv_id=arxiv_id,
        title="Attention Evolutions",
        section="Body",
        text=text,
        page=None,
        chunk_index=index,
    )


@pytest.mark.integration
async def test_index_then_bm25_retrieve_real_roundtrip() -> None:
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-test")
    await conn.create_index()

    indexer = ChunkIndexer(connection=conn, embedder=FakeEmbedder(dimension=16))
    chunks = [
        _chunk(0, "Transformers use self-attention mechanisms."),
        _chunk(1, "Recurrent networks process sequences step by step."),
    ]
    count = await indexer.index_chunks(chunks)
    assert count == 2

    retriever = BM25Retriever(conn)
    results = await retriever.search("attention", top_k=10)

    assert len(results) == 1
    assert results[0].chunk_id == "p1:chunk:0000"


@pytest.mark.integration
async def test_index_then_vector_retrieve_real_roundtrip() -> None:
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-test")
    await conn.create_index()

    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)
    chunks = [_chunk(0, "Transformers use self-attention."), _chunk(1, "Other unrelated text.")]
    await indexer.index_chunks(chunks)

    retriever = VectorRetriever(conn)
    query_vector = await embedder.embed_query("attention mechanisms")
    results = await retriever.search(query_vector, top_k=10)

    assert len(results) == 2  # fake kNN returns all docs
    assert all(r.retrieval_method == "vector" for r in results)


@pytest.mark.integration
async def test_full_hybrid_retrieval_service_real_roundtrip() -> None:
    """The real integration: index via ChunkIndexer, then retrieve via
    RetrievalService -> HybridRetriever -> BM25 + Vector -> RRF, all real
    objects, only the OpenSearch transport is faked.
    """
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-test")
    await conn.create_index()

    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)
    chunks = [
        _chunk(0, "Attention mechanisms revolutionized NLP."),
        _chunk(1, "Convolutional networks are used in vision."),
        _chunk(2, "Attention is also used in vision transformers."),
    ]
    await indexer.index_chunks(chunks)

    service = RetrievalService(
        hybrid_retriever=HybridRetriever(
            bm25_retriever=BM25Retriever(conn),
            vector_retriever=VectorRetriever(conn),
            embedder=embedder,
        )
    )

    results = await service.retrieve("attention", top_k=10)

    result_ids = {r.chunk_id for r in results}
    assert "p1:chunk:0000" in result_ids  # matched by BM25 (text contains "attention")
    assert "p1:chunk:0002" in result_ids
    assert all(r.retrieval_method == "hybrid_rrf" for r in results)
