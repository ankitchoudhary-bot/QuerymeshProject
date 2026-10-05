"""Integration test: `ResponseCache` genuinely wired into
`run_research_workflow`/`run_research_workflow_streaming` — the "complete
final end-to-end request flow" Phase 19 left open.

`test_cache_integration.py` (pre-existing) proves `ResponseCache` itself
works; it never calls `run_research_workflow(..., cache=...)` at all. This
file proves the *wiring*: a cache hit must skip the graph entirely (no
LLM/retrieval call happens a second time), not just that get/set work in
isolation. An `_CountingLLM` makes that provable rather than assumed —
if the graph secretly still ran on a cache hit, `call_count` would catch it.
"""

from __future__ import annotations

import json

import pytest

from querymesh.cache import ResponseCache
from querymesh.embeddings.fake import FakeEmbedder
from querymesh.graph import (
    compile_workflow,
    create_initial_state,
    run_research_workflow,
    run_research_workflow_streaming,
)
from querymesh.graph.stream_events import WORKFLOW_COMPLETED, WORKFLOW_STARTED
from querymesh.indexing.indexer import ChunkIndexer
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.llm.base import LLMChunk, LLMResponse
from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever
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


class _CountingLLM:
    """Returns each canned response in order (repeating the last once
    exhausted) and counts every real invocation — the proof a cache hit
    never touches the graph.
    """

    model = "counting"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0
        self.call_count = 0

    async def ainvoke(self, prompt: str) -> LLMResponse:
        self.call_count += 1
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return LLMResponse(text=text, model=self.model)

    async def astream(self, prompt: str):
        self.call_count += 1
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        yield LLMChunk(text=text, model=self.model, is_final=True)


async def _build_compiled_graph(llm):
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-cache-workflow")
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
                text="Attention mechanisms evolved over time.",
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
    return compile_workflow(llm=llm, retrieval_service=retrieval_service)  # type: ignore[arg-type]


def _happy_path_responses(answer: str) -> list[str]:
    return [
        json.dumps({"search_queries": ["attention"], "reasoning": "plan"}),  # research
        json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}),  # grade
        answer,  # synthesize
        json.dumps({"mappings": [{"sentence_index": 0, "chunk_ids": ["p1:chunk:0000"]}]}),  # citation
    ]


# --- run_research_workflow: real cache hit skips the graph -----------------


@pytest.mark.integration
async def test_second_identical_call_skips_the_graph_entirely() -> None:
    llm = _CountingLLM(_happy_path_responses("Attention mechanisms evolved from full to sparse forms."))
    compiled = await _build_compiled_graph(llm)
    cache = ResponseCache(client=_FakeRedisClient())  # type: ignore[arg-type]
    query = "How have attention mechanisms evolved?"

    first = await run_research_workflow(compiled, create_initial_state(query, max_retries=1), cache=cache)
    calls_after_first_run = llm.call_count
    assert calls_after_first_run > 0
    assert first["metadata"].get("cache_hit") is not True

    second = await run_research_workflow(compiled, create_initial_state(query, max_retries=1), cache=cache)

    assert llm.call_count == calls_after_first_run, "a cache hit must never invoke the LLM again"
    assert second["answer"] == first["answer"]
    assert second["metadata"]["cache_hit"] is True
    assert [c.chunk_id for c in second["citations"]] == [c.chunk_id for c in first["citations"]]


@pytest.mark.integration
async def test_cache_hit_preserves_retrieved_document_count_in_metadata() -> None:
    """The reconstructed cache-hit result can't replay real evidence (the
    cache never stored it) but should still surface the original
    retrieved_document_count so a caller can tell the two runs apart.

    Note: `retrieved_document_count` is computed for what gets *stored* in
    Redis (`_cache_metadata`) -- it's never merged back into the live
    result of the run that produced it, only reconstructed on a later
    cache hit (`_result_from_cache`). So this is checked on `second`
    (the hit) and directly against the cache, not on `first`.
    """
    llm = _CountingLLM(_happy_path_responses("An answer."))
    compiled = await _build_compiled_graph(llm)
    cache = ResponseCache(client=_FakeRedisClient())  # type: ignore[arg-type]
    query = "How have attention mechanisms evolved?"

    first = await run_research_workflow(compiled, create_initial_state(query, max_retries=1), cache=cache)
    cached_directly = await cache.get(query)
    second = await run_research_workflow(compiled, create_initial_state(query, max_retries=1), cache=cache)

    assert first["retrieved_documents"]  # the real run actually retrieved something
    assert cached_directly["metadata"]["retrieved_document_count"] >= 1
    assert second["metadata"]["retrieved_document_count"] == cached_directly["metadata"]["retrieved_document_count"]
    assert second["retrieved_documents"] == []  # documented trade-off: not replayed on a hit


@pytest.mark.integration
async def test_empty_answer_is_never_cached() -> None:
    """Per ResponseCache.set()'s own rule: a failed/empty run must not
    poison the cache for the next attempt."""
    llm = _CountingLLM(
        [
            json.dumps({"search_queries": ["quantum computing"], "reasoning": "plan"}),
            json.dumps({"relevant": False, "score": 0.0, "reason": "off topic"}),
            "",  # empty synthesis -- nothing to cache
        ]
    )
    compiled = await _build_compiled_graph(llm)
    cache = ResponseCache(client=_FakeRedisClient())  # type: ignore[arg-type]
    query = "What is quantum computing?"

    result = await run_research_workflow(compiled, create_initial_state(query, max_retries=0), cache=cache)

    assert result["answer"] == ""
    assert await cache.exists(query) is False


@pytest.mark.integration
async def test_without_cache_every_call_reruns_the_graph() -> None:
    """cache=None (the default) -> genuinely opt-in, never a silent
    always-on shortcut."""
    llm = _CountingLLM(_happy_path_responses("Attention mechanisms evolved from full to sparse forms."))
    compiled = await _build_compiled_graph(llm)
    query = "How have attention mechanisms evolved?"

    await run_research_workflow(compiled, create_initial_state(query, max_retries=1))
    calls_after_first_run = llm.call_count
    await run_research_workflow(compiled, create_initial_state(query, max_retries=1))

    assert llm.call_count > calls_after_first_run


# --- run_research_workflow_streaming: real cache hit short-circuits --------


@pytest.mark.integration
async def test_streaming_second_identical_call_short_circuits_with_no_node_events() -> None:
    llm = _CountingLLM(_happy_path_responses("Attention mechanisms evolved from full to sparse forms."))
    compiled = await _build_compiled_graph(llm)
    cache = ResponseCache(client=_FakeRedisClient())  # type: ignore[arg-type]
    query = "How have attention mechanisms evolved?"
    state_factory = lambda: create_initial_state(query, max_retries=1)  # noqa: E731

    first_events = [e async for e in run_research_workflow_streaming(compiled, state_factory(), cache=cache)]
    calls_after_first_run = llm.call_count
    assert calls_after_first_run > 0
    assert first_events[0]["type"] == WORKFLOW_STARTED
    assert first_events[-1]["type"] == WORKFLOW_COMPLETED
    assert first_events[-1]["data"]["cache_hit"] is False
    assert len(first_events) > 2  # real run: per-node events in between

    second_events = [e async for e in run_research_workflow_streaming(compiled, state_factory(), cache=cache)]

    assert llm.call_count == calls_after_first_run, "a cache hit must never invoke the LLM again"
    # Short-circuit: exactly workflow_started -> workflow_completed, nothing
    # in between (no agent_started/token/citation -- no node ever ran).
    assert len(second_events) == 2
    assert second_events[0]["type"] == WORKFLOW_STARTED
    assert second_events[1]["type"] == WORKFLOW_COMPLETED
    assert second_events[1]["data"]["cache_hit"] is True
    assert second_events[1]["data"]["answer"] == first_events[-1]["data"]["answer"]
    assert second_events[1]["data"]["citations"] == first_events[-1]["data"]["citations"]


@pytest.mark.integration
async def test_streaming_without_cache_never_short_circuits() -> None:
    """Two logically-independent requests (no cache involved) must each
    run the graph fully. Each gets its own `_CountingLLM`/compiled graph
    — sharing one mutable LLM instance's response-index counter across
    two separate calls would understate how many real calls a single
    full pass needs (research + per-doc grading + synthesis, and more if
    a retry fires), silently exhausting the canned response list partway
    through and feeding garbled, wrong-shaped JSON to whichever agent
    called next. That's a real failure mode this test must not reproduce
    just to make a shortcut assertion.
    """
    query = "How have attention mechanisms evolved?"
    state_factory = lambda: create_initial_state(query, max_retries=1)  # noqa: E731

    llm_one = _CountingLLM(_happy_path_responses("An answer."))
    compiled_one = await _build_compiled_graph(llm_one)
    first_events = [e async for e in run_research_workflow_streaming(compiled_one, state_factory())]

    llm_two = _CountingLLM(_happy_path_responses("An answer."))
    compiled_two = await _build_compiled_graph(llm_two)
    second_events = [e async for e in run_research_workflow_streaming(compiled_two, state_factory())]

    assert len(second_events) == len(first_events) > 2


# --- Cache backend failures degrade gracefully, never crash the request ----


class _BrokenRedisClient:
    """Every call raises -- simulates Redis being unreachable. Real bug
    this regression-tests: `_cache_get_safe`/`_cache_set_safe` referenced
    a `logger` that was never imported in `workflow.py`, so the very
    "degrade gracefully on cache failure" path raised `NameError` instead
    of degrading -- caught by actually running this scenario, not by
    reading the code.
    """

    async def get(self, key: str):
        raise ConnectionError("redis unreachable")

    async def set(self, key: str, value: str, ex=None):
        raise ConnectionError("redis unreachable")

    async def delete(self, key: str):
        raise ConnectionError("redis unreachable")

    async def exists(self, key: str):
        raise ConnectionError("redis unreachable")

    async def close(self):
        pass


@pytest.mark.integration
async def test_cache_get_failure_degrades_to_running_the_real_graph() -> None:
    llm = _CountingLLM(_happy_path_responses("An answer despite a broken cache."))
    compiled = await _build_compiled_graph(llm)
    cache = ResponseCache(client=_BrokenRedisClient())  # type: ignore[arg-type]
    query = "How have attention mechanisms evolved?"

    result = await run_research_workflow(compiled, create_initial_state(query, max_retries=1), cache=cache)

    assert result["answer"] == "An answer despite a broken cache."
    assert llm.call_count > 0
    assert result["metadata"].get("cache_hit") is not True


@pytest.mark.integration
async def test_cache_set_failure_does_not_fail_an_otherwise_successful_request() -> None:
    """A write failure happens *after* the real answer is already
    computed -- the caller must still get it back, not an exception.
    """
    llm = _CountingLLM(_happy_path_responses("An answer that fails to cache."))
    compiled = await _build_compiled_graph(llm)
    cache = ResponseCache(client=_BrokenRedisClient())  # type: ignore[arg-type]
    query = "How have attention mechanisms evolved?"

    result = await run_research_workflow(compiled, create_initial_state(query, max_retries=1), cache=cache)

    assert result["answer"] == "An answer that fails to cache."


@pytest.mark.integration
async def test_streaming_cache_get_failure_degrades_to_running_the_real_graph() -> None:
    llm = _CountingLLM(_happy_path_responses("An answer despite a broken cache."))
    compiled = await _build_compiled_graph(llm)
    cache = ResponseCache(client=_BrokenRedisClient())  # type: ignore[arg-type]
    query = "How have attention mechanisms evolved?"

    events = [
        e
        async for e in run_research_workflow_streaming(
            compiled, create_initial_state(query, max_retries=1), cache=cache
        )
    ]

    assert events[0]["type"] == WORKFLOW_STARTED
    assert events[-1]["type"] == WORKFLOW_COMPLETED
    assert events[-1]["data"]["answer"] == "An answer despite a broken cache."
    assert events[-1]["data"]["cache_hit"] is False
    assert len(events) > 2  # a real graph run happened, not a (broken) short-circuit
    assert llm.call_count > 0


@pytest.mark.integration
async def test_streaming_cache_set_failure_does_not_crash_the_stream() -> None:
    llm = _CountingLLM(_happy_path_responses("An answer that fails to cache."))
    compiled = await _build_compiled_graph(llm)
    cache = ResponseCache(client=_BrokenRedisClient())  # type: ignore[arg-type]
    query = "How have attention mechanisms evolved?"

    events = [
        e
        async for e in run_research_workflow_streaming(
            compiled, create_initial_state(query, max_retries=1), cache=cache
        )
    ]

    assert events[-1]["type"] == WORKFLOW_COMPLETED
    assert events[-1]["data"]["answer"] == "An answer that fails to cache."
