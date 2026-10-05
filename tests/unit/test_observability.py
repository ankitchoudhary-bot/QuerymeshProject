"""Phase 18 tests: Tracer abstraction (NoOp/Logfire/Langfuse/Composite),
scrub_attributes, the configure_* no-op-without-credentials pattern, the
get_tracer factory, and the tracing hierarchy actually firing through a
real graph run.
"""

from __future__ import annotations

import json

import pytest

from querymesh.config import Settings, get_settings
from querymesh.observability import CompositeTracer, NoOpTracer, get_tracer, scrub_attributes
from querymesh.observability.langfuse_setup import configure_langfuse
from querymesh.observability.logfire_setup import configure_logfire
from querymesh.observability.tracing import LangfuseTracer, LogfireTracer, Tracer


# --- scrub_attributes -----------------------------------------------------


@pytest.mark.unit
def test_scrub_attributes_drops_sensitive_keys() -> None:
    attrs = {
        "query": "attention mechanisms",
        "api_key": "sk-secret",
        "jina_api_key": "sk-secret2",
        "auth_token": "tok",
        "password": "hunter2",
        "top_k": 10,
    }
    scrubbed = scrub_attributes(attrs)

    assert scrubbed == {"query": "attention mechanisms", "top_k": 10}


@pytest.mark.unit
def test_scrub_attributes_keeps_ordinary_data() -> None:
    attrs = {"query": "q", "document_count": 5, "cache_hit": True}
    assert scrub_attributes(attrs) == attrs


# --- NoOpTracer -----------------------------------------------------------


@pytest.mark.unit
async def test_noop_tracer_span_does_nothing_and_never_raises() -> None:
    tracer: Tracer = NoOpTracer()
    with tracer.span("research_request", query="q", api_key="should-be-fine-here-too"):
        pass  # must not raise


# --- configure_logfire / configure_langfuse: no-op without credentials ----


@pytest.mark.unit
def test_configure_logfire_returns_none_without_token() -> None:
    settings = Settings(LOGFIRE_TOKEN=None)
    assert configure_logfire(settings) is None


@pytest.mark.unit
def test_configure_logfire_returns_module_with_token() -> None:
    settings = Settings(LOGFIRE_TOKEN="fake-token-for-config-only")
    logfire_module = configure_logfire(settings)
    assert logfire_module is not None
    assert hasattr(logfire_module, "span")


@pytest.mark.unit
def test_configure_langfuse_returns_none_without_credentials() -> None:
    settings = Settings(LANGFUSE_PUBLIC_KEY=None, LANGFUSE_SECRET_KEY=None)
    assert configure_langfuse(settings) is None


@pytest.mark.unit
def test_configure_langfuse_returns_none_with_only_one_key() -> None:
    settings = Settings(LANGFUSE_PUBLIC_KEY="pub", LANGFUSE_SECRET_KEY=None)
    assert configure_langfuse(settings) is None


@pytest.mark.unit
def test_configure_langfuse_returns_client_with_both_keys() -> None:
    settings = Settings(LANGFUSE_PUBLIC_KEY="pub", LANGFUSE_SECRET_KEY="secret")
    client = configure_langfuse(settings)
    assert client is not None
    assert hasattr(client, "start_as_current_observation")


# --- get_tracer factory --------------------------------------------------


@pytest.mark.unit
def test_get_tracer_defaults_to_noop_without_any_credentials() -> None:
    settings = Settings(LOGFIRE_TOKEN=None, LANGFUSE_PUBLIC_KEY=None, LANGFUSE_SECRET_KEY=None)
    tracer = get_tracer(settings)
    assert isinstance(tracer, NoOpTracer)


@pytest.mark.unit
def test_get_tracer_returns_logfire_tracer_with_only_logfire_configured() -> None:
    settings = Settings(
        LOGFIRE_TOKEN="fake-token", LANGFUSE_PUBLIC_KEY=None, LANGFUSE_SECRET_KEY=None
    )
    tracer = get_tracer(settings)
    assert isinstance(tracer, LogfireTracer)


@pytest.mark.unit
def test_get_tracer_returns_composite_with_both_configured() -> None:
    settings = Settings(
        LOGFIRE_TOKEN="fake-token", LANGFUSE_PUBLIC_KEY="pub", LANGFUSE_SECRET_KEY="secret"
    )
    tracer = get_tracer(settings)
    assert isinstance(tracer, CompositeTracer)


# --- CompositeTracer: fan-out and failure isolation ---------------------


class _SpyTracer:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.spans_opened: list[str] = []

    def span(self, name: str, **attributes):
        import contextlib

        @contextlib.contextmanager
        def _cm():
            if self.fail:
                raise RuntimeError(f"{self.name} backend unavailable")
            self.spans_opened.append(name)
            yield

        return _cm()


@pytest.mark.unit
def test_composite_tracer_fans_out_to_all_children() -> None:
    spy_a = _SpyTracer("a")
    spy_b = _SpyTracer("b")
    composite = CompositeTracer([spy_a, spy_b])

    with composite.span("research_request", query="q"):
        pass

    assert spy_a.spans_opened == ["research_request"]
    assert spy_b.spans_opened == ["research_request"]


@pytest.mark.unit
def test_composite_tracer_isolates_one_backend_failure() -> None:
    working = _SpyTracer("working")
    broken = _SpyTracer("broken", fail=True)
    composite = CompositeTracer([broken, working])

    # Must not raise even though one backend's span() raises.
    with composite.span("research_request"):
        pass

    assert working.spans_opened == ["research_request"]


# --- Tracing hierarchy actually fires through a real graph run -----------


@pytest.mark.unit
async def test_tracing_hierarchy_fires_through_full_graph_run() -> None:
    """End-to-end: research_request -> researcher/retrieval(bm25/vector/rrf)
    /grader/synthesis/citation all actually open, in the right nesting.
    """
    from querymesh.embeddings.fake import FakeEmbedder
    from querymesh.graph import compile_workflow, create_initial_state, run_research_workflow
    from querymesh.indexing.opensearch import OpenSearchConnection
    from querymesh.llm.fake import FakeLLM
    from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever

    class _FakeConnection:
        async def search(self, body: dict) -> dict:
            hit = {
                "_id": "a:chunk:0000",
                "_score": 1.0,
                "_source": {"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "t"},
            }
            return {"hits": {"hits": [hit]}}

    spy = _SpyTracer("spy")
    conn = _FakeConnection()
    hybrid = HybridRetriever(
        bm25_retriever=BM25Retriever(conn),  # type: ignore[arg-type]
        vector_retriever=VectorRetriever(conn),  # type: ignore[arg-type]
        embedder=FakeEmbedder(dimension=8),
        tracer=spy,  # type: ignore[arg-type]
    )
    retrieval_service = RetrievalService(hybrid_retriever=hybrid, tracer=spy)  # type: ignore[arg-type]

    llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}))
    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service, tracer=spy)  # type: ignore[arg-type]

    await run_research_workflow(compiled, create_initial_state("q", max_retries=1), tracer=spy)  # type: ignore[arg-type]

    assert spy.spans_opened == [
        "research_request",
        "researcher",
        "retrieval",
        "bm25",
        "vector",
        "rrf",
        "grader",
        "synthesis",
        "citation",
    ]
