"""Phase 8 tests: VectorRetriever (k-NN query construction/parsing) and
ChunkIndexer (embed + bulk-index), both against fake connections/embedders
so no live cluster or Jina credentials are needed.
"""

from __future__ import annotations

import pytest

from querymesh.embeddings.fake import FakeEmbedder
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.retrieval import RetrievedDocument, VectorRetriever
from querymesh.indexing.indexer import ChunkIndexer


class _FakeSearchConnection:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.last_body: dict | None = None

    async def search(self, body: dict) -> dict:
        self.last_body = body
        size = body.get("size", len(self._hits))
        return {"hits": {"hits": self._hits[:size]}}


class _FakeIndexConnection:
    def __init__(self) -> None:
        self.bulk_calls: list[list[dict]] = []

    async def bulk_index(self, documents: list[dict], id_field: str = "chunk_id") -> int:
        self.bulk_calls.append(documents)
        return len(documents)


def _hit(chunk_id: str, score: float) -> dict:
    return {"_id": chunk_id, "_score": score, "_source": {"chunk_id": chunk_id, "text": "x"}}


def _chunk(index: int) -> PaperChunk:
    return PaperChunk(
        chunk_id=f"a:chunk:{index:04d}",
        arxiv_id="a",
        title="Paper",
        section="Body",
        text=f"chunk text {index}",
        page=None,
        chunk_index=index,
    )


# --- VectorRetriever ------------------------------------------------------


@pytest.mark.unit
async def test_vector_search_builds_knn_query() -> None:
    conn = _FakeSearchConnection(hits=[])
    retriever = VectorRetriever(connection=conn)  # type: ignore[arg-type]

    query_vector = [0.1, 0.2, 0.3]
    await retriever.search(query_vector, top_k=5)

    assert conn.last_body is not None
    assert conn.last_body["size"] == 5
    knn = conn.last_body["query"]["knn"]["embedding"]
    assert knn["vector"] == query_vector
    assert knn["k"] == 5


@pytest.mark.unit
async def test_vector_search_parses_hits_into_retrieved_documents() -> None:
    hits = [_hit("a:chunk:0000", score=0.98), _hit("a:chunk:0001", score=0.81)]
    conn = _FakeSearchConnection(hits=hits)
    retriever = VectorRetriever(connection=conn)  # type: ignore[arg-type]

    results = await retriever.search([0.1] * 8, top_k=10)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedDocument) for r in results)
    assert results[0].retrieval_method == "vector"
    assert results[0].rank == 1
    assert results[0].chunk_id == "a:chunk:0000"


@pytest.mark.unit
async def test_vector_search_custom_field_name() -> None:
    conn = _FakeSearchConnection(hits=[])
    retriever = VectorRetriever(connection=conn, vector_field="custom_vec")  # type: ignore[arg-type]

    await retriever.search([0.1, 0.2], top_k=3)

    assert "custom_vec" in conn.last_body["query"]["knn"]


@pytest.mark.unit
async def test_vector_search_handles_no_hits() -> None:
    conn = _FakeSearchConnection(hits=[])
    retriever = VectorRetriever(connection=conn)  # type: ignore[arg-type]
    results = await retriever.search([0.1, 0.2], top_k=10)
    assert results == []


# --- ChunkIndexer -----------------------------------------------------------


@pytest.mark.unit
async def test_chunk_indexer_embeds_and_bulk_indexes() -> None:
    conn = _FakeIndexConnection()
    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)  # type: ignore[arg-type]

    chunks = [_chunk(0), _chunk(1)]
    count = await indexer.index_chunks(chunks)

    assert count == 2
    assert len(conn.bulk_calls) == 1
    indexed_docs = conn.bulk_calls[0]
    assert len(indexed_docs) == 2
    assert all(len(doc["embedding"]) == 16 for doc in indexed_docs)
    assert indexed_docs[0]["chunk_id"] == "a:chunk:0000"


@pytest.mark.unit
async def test_chunk_indexer_empty_list_is_noop() -> None:
    conn = _FakeIndexConnection()
    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)  # type: ignore[arg-type]

    count = await indexer.index_chunks([])

    assert count == 0
    assert conn.bulk_calls == []


@pytest.mark.unit
async def test_chunk_indexer_raises_on_embedding_count_mismatch() -> None:
    conn = _FakeIndexConnection()

    class _BrokenEmbedder:
        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2]]  # always returns 1, regardless of input size

        async def embed_query(self, query: str) -> list[float]:
            return [0.1, 0.2]

    indexer = ChunkIndexer(connection=conn, embedder=_BrokenEmbedder())  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        await indexer.index_chunks([_chunk(0), _chunk(1)])
