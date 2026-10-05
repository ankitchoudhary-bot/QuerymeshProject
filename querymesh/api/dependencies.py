"""API dependencies: request/correlation ID extraction and app-state access.

`AppState` holds the long-lived objects the API needs per request (the
compiled graph, tracer, response cache) — built once at startup in
`app.py`'s lifespan handler, not reconstructed per request.
"""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph

from querymesh.cache import ResponseCache
from querymesh.observability.tracing import NoOpTracer, Tracer

REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"

# Set once per request by the ASGI-level `RequestIDMiddleware` (app.py) so
# every dependency/handler that needs the request's ID sees the SAME
# value, whether it reads it from a `Request` (route/dependency code) or
# has no `Request` object at all (e.g. deep inside error handling). Using
# a contextvar rather than only `request.headers` fixes a real bug: without
# it, the middleware and a route handler each independently generated a
# *different* random UUID whenever the client didn't supply one, so the
# response header and the response body disagreed.
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "querymesh_request_id", default=None
)
_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "querymesh_correlation_id", default=None
)


@dataclass
class AppState:
    compiled_graph: CompiledStateGraph | None
    tracer: Tracer
    cache: ResponseCache | None
    query_timeout_seconds: float = 60.0


def get_app_state(request: Request) -> AppState:
    return request.app.state.querymesh


def get_or_create_request_id(request: Request) -> str:
    ctx_value = _request_id_var.get()
    if ctx_value:
        return ctx_value
    return request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())


def get_or_create_correlation_id(request: Request) -> str:
    ctx_value = _correlation_id_var.get()
    if ctx_value:
        return ctx_value
    # Correlation ID ties a chain of requests together (e.g. a client
    # retry, or a request that fans out to other services); request ID
    # identifies this one specific request. Default correlation_id to the
    # request_id when the caller doesn't supply one — a request is always
    # at least correlated with itself.
    return request.headers.get(CORRELATION_ID_HEADER) or get_or_create_request_id(request)


def default_app_state() -> AppState:
    """Build the real, production `AppState` from configured factories.

    Called once at startup (see `app.py`'s lifespan). Fails fast with a
    clear `ConfigurationError`/`IndexingError` if required config (e.g.
    `OPENSEARCH_URL`) is missing — correct behavior for a production API,
    unlike the Fake-fallback pattern used by individual component
    factories for local development.
    """
    from querymesh.embeddings import get_embedder
    from querymesh.graph import compile_workflow
    from querymesh.indexing import OpenSearchConnection, build_client
    from querymesh.llm import get_llm
    from querymesh.observability import get_tracer
    from querymesh.retrieval import (
        BM25Retriever,
        HybridRetriever,
        RetrievalService,
        VectorRetriever,
    )
    from querymesh.config import get_settings

    settings = get_settings()
    tracer = get_tracer()
    llm = get_llm()
    client = build_client()
    connection = OpenSearchConnection(client=client, index_name="querymesh-papers")
    embedder = get_embedder()

    retrieval_service = RetrievalService(
        hybrid_retriever=HybridRetriever(
            bm25_retriever=BM25Retriever(connection),
            vector_retriever=VectorRetriever(connection),
            embedder=embedder,
            tracer=tracer,
        ),
        tracer=tracer,
    )
    compiled_graph = compile_workflow(llm=llm, retrieval_service=retrieval_service, tracer=tracer)

    cache: ResponseCache | None
    try:
        from querymesh.cache import get_response_cache

        cache = get_response_cache(tracer=tracer)
    except Exception:  # noqa: BLE001 - cache is an optional optimization, never required to start
        cache = None

    return AppState(
        compiled_graph=compiled_graph,
        tracer=tracer,
        cache=cache,
        query_timeout_seconds=settings.api_query_timeout_seconds,
    )
