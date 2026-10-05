# 

"""Real performance measurements across the pipeline's layers: retrieval,
embedding, LLM, graph, cache (hit vs miss), MCP, A2A, and end-to-end.

Per the phase rule ("measure, then optimize only obvious bottlenecks —
don't prematurely optimize"), this file MEASURES and asserts sane,
directionally-correct relationships (e.g. cache hit is faster than a full
pipeline run) — it does not attempt to tune anything. In this sandbox,
every external boundary is faked (no real network/OpenSearch/Redis/LLM
provider), so absolute numbers aren't meaningful production benchmarks;
what's real and checkable is the *relative* comparison and the fact that
every layer is independently measurable.
"""

from __future__ import annotations

import json

import pytest

from querymesh.a2a import A2AClient
from querymesh.cache import ResponseCache
from querymesh.embeddings.fake import FakeEmbedder
from querymesh.graph import compile_workflow, create_initial_state
from querymesh.indexing.indexer import ChunkIndexer
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.llm.fake import FakeLLM
from querymesh.mcp.servers import build_research_server
from querymesh.observability.timing import mean_duration_ms, measure, measure_many
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
        return 1 if self._store.pop(key, None) is not None else 0

    async def exists(self, key: str):
        return 1 if key in self._store else 0

    async def close(self):
        pass


async def _build_retrieval_service() -> tuple[RetrievalService, FakeEmbedder]:
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-perf")
    await conn.create_index()

    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)
    await indexer.index_chunks(
        [
            PaperChunk(
                chunk_id=f"p1:chunk:{i:04d}",
                arxiv_id="p1",
                title="T",
                section="Body",
                text=f"Attention mechanism content number {i}.",
                page=None,
                chunk_index=i,
            )
            for i in range(5)
        ]
    )

    service = RetrievalService(
        hybrid_retriever=HybridRetriever(
            bm25_retriever=BM25Retriever(conn),
            vector_retriever=VectorRetriever(conn),
            embedder=embedder,
        )
    )
    return service, embedder


@pytest.mark.evaluation
async def test_retrieval_latency_is_measurable() -> None:
    service, _embedder = await _build_retrieval_service()

    results = await measure_many("retrieval", lambda: service.retrieve("attention", top_k=10), repeats=5)

    assert len(results) == 5
    assert all(r.duration_ms >= 0 for r in results)
    print(f"\nretrieval mean latency: {mean_duration_ms(results):.3f}ms")


@pytest.mark.evaluation
async def test_embedding_latency_is_measurable() -> None:
    embedder = FakeEmbedder(dimension=16)

    results = await measure_many("embedding", lambda: embedder.embed_query("attention mechanisms"), repeats=5)

    assert len(results) == 5
    print(f"\nembedding mean latency: {mean_duration_ms(results):.3f}ms")


@pytest.mark.evaluation
async def test_llm_latency_is_measurable() -> None:
    llm = FakeLLM(response_text="a synthesized answer")

    results = await measure_many("llm_invoke", lambda: llm.ainvoke("What is attention?"), repeats=5)

    assert len(results) == 5
    print(f"\nllm mean latency: {mean_duration_ms(results):.3f}ms")


@pytest.mark.evaluation
async def test_graph_end_to_end_latency_is_measurable() -> None:
    service, _embedder = await _build_retrieval_service()
    llm = _ScriptedLLM(
        [
            json.dumps({"search_queries": ["attention"], "reasoning": "plan"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}),
            "The synthesized final answer.",
        ]
    )
    compiled = compile_workflow(llm=llm, retrieval_service=service)  # type: ignore[arg-type]
    state = create_initial_state("How do attention mechanisms work?", max_retries=1)

    _result, latency = await measure("end_to_end", compiled.ainvoke(state))

    assert latency.duration_ms >= 0
    print(f"\nend-to-end graph latency: {latency.duration_ms:.3f}ms")


@pytest.mark.evaluation
async def test_mcp_call_latency_is_measurable() -> None:
    from fastmcp import Client

    from querymesh.agents.researcher import ResearcherAgent

    llm = FakeLLM(response_text=json.dumps({"search_queries": ["q"]}))
    server = build_research_server(ResearcherAgent(llm))

    async def call_tool():
        async with Client(server) as client:
            return await client.call_tool("search_papers", {"query": "attention"})

    results = await measure_many("mcp_call", call_tool, repeats=3)

    assert len(results) == 3
    print(f"\nMCP call mean latency: {mean_duration_ms(results):.3f}ms")


@pytest.mark.evaluation
async def test_a2a_dispatch_latency_is_measurable() -> None:
    client = A2AClient()

    async def handler(payload: dict) -> dict:
        return {"ok": True}

    client.register_handler("worker", handler)

    results = await measure_many(
        "a2a_dispatch",
        lambda: client.send(sender="a", recipient="worker", task="t", payload={}),
        repeats=5,
    )

    assert len(results) == 5
    print(f"\nA2A dispatch mean latency: {mean_duration_ms(results):.3f}ms")


@pytest.mark.evaluation
async def test_cache_hit_is_faster_than_full_pipeline_run() -> None:
    """The one directionally-meaningful comparison this sandbox can prove:
    a cache hit (one Redis GET) is faster than re-running the full graph.
    This is the real point of caching, measured rather than assumed.
    """
    service, _embedder = await _build_retrieval_service()
    query = "How do attention mechanisms work?"

    llm = _ScriptedLLM(
        [
            json.dumps({"search_queries": ["attention"], "reasoning": "plan"}),
            *([json.dumps({"relevant": True, "score": 0.9, "reason": "ok"})] * 5),
            "The synthesized final answer.",
        ]
    )
    compiled = compile_workflow(llm=llm, retrieval_service=service)  # type: ignore[arg-type]

    # Cache miss: full graph run.
    state = create_initial_state(query, max_retries=1)
    result, miss_latency = await measure("cache_miss_full_run", compiled.ainvoke(state))

    # Populate cache with the result, then measure a hit.
    cache = ResponseCache(client=_FakeRedisClient())  # type: ignore[arg-type]
    await cache.set(query, answer=result["answer"], citations=[c.model_dump() for c in result["citations"]])

    _cached, hit_latency = await measure("cache_hit", cache.get(query))

    print(f"\ncache miss (full run): {miss_latency.duration_ms:.3f}ms")
    print(f"cache hit: {hit_latency.duration_ms:.3f}ms")

    assert hit_latency.duration_ms < miss_latency.duration_ms
