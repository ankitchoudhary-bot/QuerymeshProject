"""Integration test: ResponseCache wired to a graph run's actual output —
confirms the whole "run graph, cache the answer, look it up again" flow
works together, not just each piece independently.
"""

from __future__ import annotations

import json

import pytest

from querymesh.cache import ResponseCache, make_cache_key
from querymesh.embeddings.fake import FakeEmbedder
from querymesh.graph import compile_workflow, create_initial_state
from querymesh.indexing.indexer import ChunkIndexer
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever
from tests.integration.test_end_to_end import _ScriptedLLM
from tests.integration.test_retrieval_integration import _FakeOpenSearchClient


class _FakeRedisClient:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def get(self, key: str):
        return self._store.get(key)

    async def set(self, key: str, value: str, ex=None):
        self._store[key] = value
        return True

    async def delete(self, key: str):
        existed = key in self._store
        self._store.pop(key, None)
        return 1 if existed else 0

    async def exists(self, key: str):
        return 1 if key in self._store else 0

    async def close(self):
        pass


@pytest.mark.integration
async def test_graph_result_can_be_cached_and_retrieved() -> None:
    query = "How have attention mechanisms evolved?"

    # Real graph run producing a real answer + citations.
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-cache-e2e")
    await conn.create_index()
    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)
    await indexer.index_chunks(
        [
            PaperChunk(
                chunk_id="p1:chunk:0000",
                arxiv_id="p1",
                title="T",
                section="Body",
                text="Attention mechanisms have evolved significantly.",
                page=None,
                chunk_index=0,
            )
        ]
    )
    retrieval_service = RetrievalService(
        hybrid_retriever=HybridRetriever(
            bm25_retriever=BM25Retriever(conn),
            vector_retriever=VectorRetriever(conn),
            embedder=embedder,
        )
    )
    llm = _ScriptedLLM(
        [
            json.dumps({"search_queries": ["attention"], "reasoning": "plan"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}),
            "Attention mechanisms evolved from full to sparse forms.",
        ]
    )
    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)  # type: ignore[arg-type]
    result = await compiled.ainvoke(create_initial_state(query, max_retries=1))

    # Now cache it, exactly as a real request handler would.
    cache = ResponseCache(client=_FakeRedisClient())  # type: ignore[arg-type]

    cached_before = await cache.get(query)
    assert cached_before is None  # nothing cached yet

    citations_as_dicts = [c.model_dump() for c in result["citations"]]
    ok = await cache.set(query, answer=result["answer"], citations=citations_as_dicts)
    assert ok is True

    cached_after = await cache.get(query)
    assert cached_after["answer"] == result["answer"]
    assert cached_after["citations"] == citations_as_dicts

    # Same query, different casing/whitespace -> same cache entry.
    assert await cache.exists("  HOW have Attention   mechanisms evolved?  ") is True
    assert make_cache_key(query) == make_cache_key("  HOW have Attention   mechanisms evolved?  ")
