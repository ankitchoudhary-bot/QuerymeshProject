"""API v1 routes.

Thin translation layer only — every route calls into the existing,
already-tested `querymesh.graph` workflow rather than duplicating any
graph/agent/retrieval logic, per the phase's explicit instruction.
"""

from __future__ import annotations

import asyncio
import json
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from querymesh.api.dependencies import (
    AppState,
    get_app_state,
    get_or_create_correlation_id,
    get_or_create_request_id,
)
from querymesh.api.errors import timeout_error_response
from querymesh.api.models import (
    CitationResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
)
from querymesh.auth import TokenClaims, require_scopes
from querymesh.errors import ConfigurationError
from querymesh.graph import create_initial_state, run_research_workflow, run_research_workflow_streaming
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

QUERY_SCOPE = "query:submit"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness: is the process up at all? No dependency checks and no
    auth — a slow/unreachable OpenSearch shouldn't make Kubernetes think
    this pod needs restarting (that's what /ready is for), and the
    kubelet's liveness probe carries no Authorization header to check.
    """
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(state: AppState = Depends(get_app_state)) -> ReadyResponse:
    """Readiness: can this instance actually serve a query right now?
    Also unauthenticated — same reason as /health: Kubernetes readiness
    probes don't carry credentials.
    """
    checks = {"graph_compiled": state.compiled_graph is not None}
    status = "ready" if all(checks.values()) else "not_ready"
    return ReadyResponse(status=status, checks=checks)


@router.post("/query", response_model=QueryResponse)
async def submit_query(
    request: Request,
    body: QueryRequest,
    state: AppState = Depends(get_app_state),
    _user: TokenClaims = Depends(require_scopes(QUERY_SCOPE)),
):
    if state.compiled_graph is None:
        raise ConfigurationError("QueryMesh graph is not initialized")

    request_id = get_or_create_request_id(request)
    correlation_id = get_or_create_correlation_id(request)

    try:
        with state.tracer.span("api_query", request_id=request_id, correlation_id=correlation_id):
            initial_state = create_initial_state(body.query, max_retries=body.max_retries)
            result = await asyncio.wait_for(
                run_research_workflow(state.compiled_graph, initial_state, tracer=state.tracer),
                timeout=state.query_timeout_seconds,
            )
    except TimeoutError:
        logger.error(
            "Query timed out after %ss (request_id=%s)", state.query_timeout_seconds, request_id
        )
        return timeout_error_response(request, state.query_timeout_seconds)

    return QueryResponse(
        request_id=request_id,
        correlation_id=correlation_id,
        query=body.query,
        answer=result["answer"],
        citations=[CitationResponse(**c.model_dump()) for c in result["citations"]],
        retrieved_document_count=len(result["retrieved_documents"]),
        retry_count=result["retry_count"],
        errors=result["errors"],
    )


@router.post("/query/stream")
async def submit_query_stream(
    request: Request,
    body: QueryRequest,
    state: AppState = Depends(get_app_state),
    _user: TokenClaims = Depends(require_scopes(QUERY_SCOPE)),
) -> StreamingResponse:
    """Server-Sent Events stream of `StreamEvent`s (Phase 22), one JSON
    object per `data:` line, terminated when a `workflow_completed` or
    `error` event is sent.
    """
    if state.compiled_graph is None:
        raise ConfigurationError("QueryMesh graph is not initialized")

    request_id = get_or_create_request_id(request)
    correlation_id = get_or_create_correlation_id(request)
    initial_state = create_initial_state(body.query, max_retries=body.max_retries)

    async def event_source():
        async for event in run_research_workflow_streaming(
            state.compiled_graph, initial_state, tracer=state.tracer
        ):
            payload = {**event, "request_id": request_id, "correlation_id": correlation_id}
            yield f"data: {json.dumps(jsonable_encoder(payload))}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        # No explicit X-Request-ID/X-Correlation-ID headers here — see the
        # matching fix + comment in api/errors.py: RequestIDMiddleware
        # already adds both to every response; setting them again here
        # produced the same duplicate-header bug for the streaming path.
    )
