# """Phase 31 tests: the QueryMesh API boundary — health, readiness, query,
# streaming, structured errors, request/correlation IDs, and timeout — all
# via FastAPI's real `TestClient` (real ASGI request/response cycle, real
# lifespan handling), with a fake compiled graph injected via `AppState` so
# no real OpenSearch/LLM/Redis is needed.
# """

# from __future__ import annotations

# import json

# import pytest
# from fastapi.testclient import TestClient

# from querymesh.api.app import create_app
# from querymesh.api.dependencies import AppState
# from querymesh.errors import ValidationError
# from querymesh.graph import compile_workflow
# from querymesh.llm.fake import FakeLLM
# from querymesh.observability.tracing import NoOpTracer
# from querymesh.retrieval.models import RetrievedDocument
# from querymesh.security.limits import MAX_QUERY_LENGTH, MAX_TOP_K


# class _FakeRetrievalService:
#     def __init__(self, docs: list[RetrievedDocument] | None = None) -> None:
#         self.docs = docs if docs is not None else [
#             RetrievedDocument(
#                 document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "attention text"},
#                 score=5.0,
#                 rank=1,
#                 retrieval_method="hybrid_rrf",
#             )
#         ]

#     async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
#         return self.docs


# def _grade_json(relevant: bool = True, score: float = 0.9) -> str:
#     return json.dumps({"relevant": relevant, "score": score, "reason": "test"})


# def _make_client(
#     compiled_graph=None, query_timeout_seconds: float = 60.0, retrieval_docs=None
# ) -> TestClient:
#     if compiled_graph is None:
#         llm = FakeLLM(response_text=_grade_json())
#         compiled_graph = compile_workflow(
#             llm=llm, retrieval_service=_FakeRetrievalService(retrieval_docs)
#         )
#     state = AppState(
#         compiled_graph=compiled_graph,
#         tracer=NoOpTracer(),
#         cache=None,
#         query_timeout_seconds=query_timeout_seconds,
#     )
#     app = create_app(app_state=state)
#     # raise_server_exceptions=False: Starlette's ServerErrorMiddleware always
#     # re-raises after sending the 500 response (see its own comment: "This
#     # allows test clients to optionally raise the error within the test
#     # case"). We want the actual HTTP response our unhandled_exception_handler
#     # produced, not the raw exception — this is the documented way to get it.
#     return TestClient(app, raise_server_exceptions=False)


# # --- Health / readiness ---------------------------------------------------


# @pytest.mark.integration
# def test_health_returns_ok() -> None:
#     with _make_client() as client:
#         response = client.get("/api/v1/health")

#     assert response.status_code == 200
#     assert response.json() == {"status": "ok"}


# @pytest.mark.integration
# def test_ready_returns_ready_when_graph_compiled() -> None:
#     with _make_client() as client:
#         response = client.get("/api/v1/ready")

#     assert response.status_code == 200
#     body = response.json()
#     assert body["status"] == "ready"
#     assert body["checks"]["graph_compiled"] is True


# @pytest.mark.integration
# def test_ready_reports_not_ready_without_graph() -> None:
#     state = AppState(compiled_graph=None, tracer=NoOpTracer(), cache=None)
#     app = create_app(app_state=state)

#     with TestClient(app) as client:
#         response = client.get("/api/v1/ready")

#     assert response.json()["status"] == "not_ready"
#     assert response.json()["checks"]["graph_compiled"] is False


# # --- Valid query ------------------------------------------------------


# @pytest.mark.integration
# def test_valid_query_returns_grounded_answer() -> None:
#     with _make_client() as client:
#         response = client.post("/api/v1/query", json={"query": "What is attention?"})

#     assert response.status_code == 200
#     body = response.json()
#     assert body["query"] == "What is attention?"
#     assert body["answer"]
#     assert len(body["citations"]) == 1
#     assert body["citations"][0]["chunk_id"] == "a:chunk:0000"
#     assert body["retrieved_document_count"] == 1
#     assert body["retry_count"] == 0
#     assert body["errors"] == []


# @pytest.mark.integration
# def test_query_response_has_request_and_correlation_ids() -> None:
#     with _make_client() as client:
#         response = client.post("/api/v1/query", json={"query": "What is attention?"})

#     body = response.json()
#     assert body["request_id"]
#     assert body["correlation_id"]
#     assert response.headers["x-request-id"] == body["request_id"]
#     assert response.headers["x-correlation-id"] == body["correlation_id"]


# @pytest.mark.integration
# def test_query_honors_client_supplied_request_id() -> None:
#     with _make_client() as client:
#         response = client.post(
#             "/api/v1/query",
#             json={"query": "What is attention?"},
#             headers={"X-Request-ID": "client-supplied-id"},
#         )

#     assert response.json()["request_id"] == "client-supplied-id"
#     assert response.headers["x-request-id"] == "client-supplied-id"


# @pytest.mark.integration
# def test_query_correlation_id_defaults_to_request_id_when_absent() -> None:
#     with _make_client() as client:
#         response = client.post(
#             "/api/v1/query",
#             json={"query": "What is attention?"},
#             headers={"X-Request-ID": "req-123"},
#         )

#     body = response.json()
#     assert body["request_id"] == "req-123"
#     assert body["correlation_id"] == "req-123"


# # --- Invalid request (Pydantic-level validation) --------------------------


# @pytest.mark.integration
# def test_empty_query_rejected() -> None:
#     with _make_client() as client:
#         response = client.post("/api/v1/query", json={"query": ""})

#     assert response.status_code == 422


# @pytest.mark.integration
# def test_missing_query_field_rejected() -> None:
#     with _make_client() as client:
#         response = client.post("/api/v1/query", json={})

#     assert response.status_code == 422


# @pytest.mark.integration
# def test_oversized_query_rejected() -> None:
#     with _make_client() as client:
#         response = client.post("/api/v1/query", json={"query": "x" * (MAX_QUERY_LENGTH + 1)})

#     assert response.status_code == 422


# @pytest.mark.integration
# def test_oversized_top_k_rejected() -> None:
#     with _make_client() as client:
#         response = client.post(
#             "/api/v1/query", json={"query": "valid query", "top_k": MAX_TOP_K + 1}
#         )

#     assert response.status_code == 422


# @pytest.mark.integration
# def test_negative_max_retries_rejected() -> None:
#     with _make_client() as client:
#         response = client.post(
#             "/api/v1/query", json={"query": "valid query", "max_retries": -1}
#         )

#     assert response.status_code == 422


# # --- Graph failure ----------------------------------------------------


# @pytest.mark.integration
# def test_graph_failure_returns_structured_error() -> None:
#     class _BrokenGraph:
#         async def ainvoke(self, state, config=None):
#             raise RuntimeError("graph exploded")

#         def astream(self, *args, **kwargs):
#             raise RuntimeError("graph exploded")

#     with _make_client(compiled_graph=_BrokenGraph()) as client:
#         response = client.post("/api/v1/query", json={"query": "What is attention?"})

#     assert response.status_code == 500
#     body = response.json()
#     assert body["error"] == "InternalError"
#     assert body["retryable"] is False
#     assert "request_id" in body


# @pytest.mark.integration
# def test_uninitialized_graph_returns_configuration_error() -> None:
#     state = AppState(compiled_graph=None, tracer=NoOpTracer(), cache=None)
#     app = create_app(app_state=state)

#     with TestClient(app) as client:
#         response = client.post("/api/v1/query", json={"query": "What is attention?"})

#     assert response.status_code == 500
#     body = response.json()
#     assert body["error"] == "ConfigurationError"
#     assert body["retryable"] is False


# # --- Timeout ------------------------------------------------------------


# @pytest.mark.integration
# def test_query_timeout_returns_504() -> None:
#     with _make_client(query_timeout_seconds=0.001) as client:
#         response = client.post("/api/v1/query", json={"query": "What is attention?"})

#     assert response.status_code == 504
#     body = response.json()
#     assert body["error"] == "RequestTimeout"
#     assert body["retryable"] is True


# # --- Streaming ----------------------------------------------------------


# @pytest.mark.integration
# def test_streaming_query_returns_sse_events() -> None:
#     with _make_client() as client:
#         with client.stream(
#             "POST", "/api/v1/query/stream", json={"query": "What is attention?"}
#         ) as response:
#             assert response.status_code == 200
#             assert response.headers["content-type"].startswith("text/event-stream")

#             lines = [line for line in response.iter_lines() if line]

#     assert len(lines) > 0
#     first_event = json.loads(lines[0].removeprefix("data: "))
#     last_event = json.loads(lines[-1].removeprefix("data: "))

#     assert first_event["type"] == "workflow_started"
#     assert last_event["type"] == "workflow_completed"
#     assert "request_id" in first_event
#     assert "correlation_id" in first_event


# @pytest.mark.integration
# def test_streaming_query_includes_token_events() -> None:
#     with _make_client() as client:
#         with client.stream(
#             "POST", "/api/v1/query/stream", json={"query": "What is attention?"}
#         ) as response:
#             lines = [line for line in response.iter_lines() if line]

#     events = [json.loads(line.removeprefix("data: ")) for line in lines]
#     event_types = {e["type"] for e in events}
#     assert "token" in event_types


# # --- Validation errors surfacing through our own handler ------------------


# @pytest.mark.integration
# def test_internal_validation_error_maps_to_422() -> None:
#     """Beyond Pydantic's own field-level checks, a `ValidationError` raised
#     deeper in the pipeline (e.g. inside create_initial_state) is caught by
#     our own handler and produces the same structured shape.
#     """
#     with _make_client() as client:
#         app = client.app
#         original_state = app.state.querymesh

#         class _AlwaysInvalidGraph:
#             async def ainvoke(self, state, config=None):
#                 raise ValidationError("simulated deep validation failure")

#         app.state.querymesh = AppState(
#             compiled_graph=_AlwaysInvalidGraph(),
#             tracer=original_state.tracer,
#             cache=None,
#             query_timeout_seconds=60.0,
#         )
#         response = client.post("/api/v1/query", json={"query": "valid query"})

#     assert response.status_code == 422
#     body = response.json()
#     assert body["error"] == "ValidationError"
#     assert body["retryable"] is False


"""Phase 31 tests: the QueryMesh API boundary — health, readiness, query,
streaming, structured errors, request/correlation IDs, and timeout — all
via FastAPI's real `TestClient` (real ASGI request/response cycle, real
lifespan handling), with a fake compiled graph injected via `AppState` so
no real OpenSearch/LLM/Redis is needed.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from querymesh.auth.dependencies import get_current_user
from querymesh.auth.models import TokenClaims
from querymesh.api.app import create_app
from querymesh.api.dependencies import AppState
from querymesh.errors import ValidationError
from querymesh.graph import compile_workflow
from querymesh.llm.fake import FakeLLM
from querymesh.observability.tracing import NoOpTracer
from querymesh.retrieval.models import RetrievedDocument
from querymesh.security.limits import MAX_QUERY_LENGTH, MAX_TOP_K


class _FakeRetrievalService:
    def __init__(self, docs: list[RetrievedDocument] | None = None) -> None:
        self.docs = docs if docs is not None else [
            RetrievedDocument(
                document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "attention text"},
                score=5.0,
                rank=1,
                retrieval_method="hybrid_rrf",
            )
        ]

    async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        return self.docs


def _grade_json(relevant: bool = True, score: float = 0.9) -> str:
    return json.dumps({"relevant": relevant, "score": score, "reason": "test"})

def _install_test_auth(app) -> None:
    async def _test_current_user() -> TokenClaims:
        return TokenClaims(
            subject="integration-test-user",
            scopes=["query:submit"],
            issuer=None,
            audience=None,
            expires_at=None,
            raw={},
        )

    app.dependency_overrides[get_current_user] = _test_current_user

def _make_client(
    compiled_graph=None, query_timeout_seconds: float = 60.0, retrieval_docs=None
) -> TestClient:
    if compiled_graph is None:
        llm = FakeLLM(response_text=_grade_json())
        compiled_graph = compile_workflow(
            llm=llm, retrieval_service=_FakeRetrievalService(retrieval_docs)
        )
    state = AppState(
        compiled_graph=compiled_graph,
        tracer=NoOpTracer(),
        cache=None,
        query_timeout_seconds=query_timeout_seconds,
    )
    app = create_app(app_state=state)
    _install_test_auth(app)

    # raise_server_exceptions=False: Starlette's ServerErrorMiddleware always
    # re-raises after sending the 500 response (see its own comment: "This
    # allows test clients to optionally raise the error within the test
    # case"). We want the actual HTTP response our unhandled_exception_handler
    # produced, not the raw exception — this is the documented way to get it.
    return TestClient(app, raise_server_exceptions=False)


# --- Health / readiness ---------------------------------------------------


@pytest.mark.integration
def test_health_returns_ok() -> None:
    with _make_client() as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_ready_returns_ready_when_graph_compiled() -> None:
    with _make_client() as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["graph_compiled"] is True


@pytest.mark.integration
def test_ready_reports_not_ready_without_graph() -> None:
    state = AppState(compiled_graph=None, tracer=NoOpTracer(), cache=None)
    app = create_app(app_state=state)

    with TestClient(app) as client:
        response = client.get("/api/v1/ready")

    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["graph_compiled"] is False


# --- Valid query ------------------------------------------------------


@pytest.mark.integration
def test_valid_query_returns_grounded_answer() -> None:
    with _make_client() as client:
        response = client.post("/api/v1/query", json={"query": "What is attention?"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "What is attention?"
    assert body["answer"]
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == "a:chunk:0000"
    assert body["retrieved_document_count"] == 1
    assert body["retry_count"] == 0
    assert body["errors"] == []


@pytest.mark.integration
def test_query_response_has_request_and_correlation_ids() -> None:
    with _make_client() as client:
        response = client.post("/api/v1/query", json={"query": "What is attention?"})

    body = response.json()
    assert body["request_id"]
    assert body["correlation_id"]
    assert response.headers["x-request-id"] == body["request_id"]
    assert response.headers["x-correlation-id"] == body["correlation_id"]


@pytest.mark.integration
def test_query_honors_client_supplied_request_id() -> None:
    with _make_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "What is attention?"},
            headers={"X-Request-ID": "client-supplied-id"},
        )

    assert response.json()["request_id"] == "client-supplied-id"
    assert response.headers["x-request-id"] == "client-supplied-id"


@pytest.mark.integration
def test_query_correlation_id_defaults_to_request_id_when_absent() -> None:
    with _make_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "What is attention?"},
            headers={"X-Request-ID": "req-123"},
        )

    body = response.json()
    assert body["request_id"] == "req-123"
    assert body["correlation_id"] == "req-123"


# --- Invalid request (Pydantic-level validation) --------------------------


@pytest.mark.integration
def test_empty_query_rejected() -> None:
    with _make_client() as client:
        response = client.post("/api/v1/query", json={"query": ""})

    assert response.status_code == 422


@pytest.mark.integration
def test_missing_query_field_rejected() -> None:
    with _make_client() as client:
        response = client.post("/api/v1/query", json={})

    assert response.status_code == 422


@pytest.mark.integration
def test_oversized_query_rejected() -> None:
    with _make_client() as client:
        response = client.post("/api/v1/query", json={"query": "x" * (MAX_QUERY_LENGTH + 1)})

    assert response.status_code == 422


@pytest.mark.integration
def test_oversized_top_k_rejected() -> None:
    with _make_client() as client:
        response = client.post(
            "/api/v1/query", json={"query": "valid query", "top_k": MAX_TOP_K + 1}
        )

    assert response.status_code == 422


@pytest.mark.integration
def test_negative_max_retries_rejected() -> None:
    with _make_client() as client:
        response = client.post(
            "/api/v1/query", json={"query": "valid query", "max_retries": -1}
        )

    assert response.status_code == 422


# --- Graph failure ----------------------------------------------------


@pytest.mark.integration
def test_graph_failure_returns_structured_error() -> None:
    class _BrokenGraph:
        async def ainvoke(self, state, config=None):
            raise RuntimeError("graph exploded")

        def astream(self, *args, **kwargs):
            raise RuntimeError("graph exploded")

    with _make_client(compiled_graph=_BrokenGraph()) as client:
        response = client.post("/api/v1/query", json={"query": "What is attention?"})

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "InternalError"
    assert body["retryable"] is False
    assert "request_id" in body


@pytest.mark.integration
def test_uninitialized_graph_returns_configuration_error() -> None:
    state = AppState(compiled_graph=None, tracer=NoOpTracer(), cache=None)
    app = create_app(app_state=state)
    _install_test_auth(app)
    
    with TestClient(app) as client:
        response = client.post("/api/v1/query", json={"query": "What is attention?"})

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "ConfigurationError"
    assert body["retryable"] is False


# --- Timeout ------------------------------------------------------------


@pytest.mark.integration
def test_query_timeout_returns_504() -> None:
    with _make_client(query_timeout_seconds=0.001) as client:
        response = client.post("/api/v1/query", json={"query": "What is attention?"})

    assert response.status_code == 504
    body = response.json()
    assert body["error"] == "RequestTimeout"
    assert body["retryable"] is True


# --- Streaming ----------------------------------------------------------


@pytest.mark.integration
def test_streaming_query_returns_sse_events() -> None:
    with _make_client() as client:
        with client.stream(
            "POST", "/api/v1/query/stream", json={"query": "What is attention?"}
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            lines = [line for line in response.iter_lines() if line]

    assert len(lines) > 0
    first_event = json.loads(lines[0].removeprefix("data: "))
    last_event = json.loads(lines[-1].removeprefix("data: "))

    assert first_event["type"] == "workflow_started"
    assert last_event["type"] == "workflow_completed"
    assert "request_id" in first_event
    assert "correlation_id" in first_event


@pytest.mark.integration
def test_streaming_query_includes_token_events() -> None:
    with _make_client() as client:
        with client.stream(
            "POST", "/api/v1/query/stream", json={"query": "What is attention?"}
        ) as response:
            lines = [line for line in response.iter_lines() if line]

    events = [json.loads(line.removeprefix("data: ")) for line in lines]
    event_types = {e["type"] for e in events}
    assert "token" in event_types


# --- Validation errors surfacing through our own handler ------------------


@pytest.mark.integration
def test_internal_validation_error_maps_to_422() -> None:
    """Beyond Pydantic's own field-level checks, a `ValidationError` raised
    deeper in the pipeline (e.g. inside create_initial_state) is caught by
    our own handler and produces the same structured shape.
    """
    with _make_client() as client:
        app = client.app
        original_state = app.state.querymesh

        class _AlwaysInvalidGraph:
            async def ainvoke(self, state, config=None):
                raise ValidationError("simulated deep validation failure")

        app.state.querymesh = AppState(
            compiled_graph=_AlwaysInvalidGraph(),
            tracer=original_state.tracer,
            cache=None,
            query_timeout_seconds=60.0,
        )
        response = client.post("/api/v1/query", json={"query": "valid query"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "ValidationError"
    assert body["retryable"] is False
