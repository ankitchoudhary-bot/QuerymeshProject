"""Phase 32 integration tests: JWT auth enforced on the real API via
FastAPI's `TestClient` — missing/malformed/expired/invalid-signature/valid
tokens, and insufficient permissions, against the actual protected routes
(`/query`, `/query/stream`), with `/health` confirmed to stay open.
"""

from __future__ import annotations

import json
import time

import jwt as pyjwt
import pytest
from fastapi.testclient import TestClient

from querymesh.api.app import create_app
from querymesh.api.dependencies import AppState
from querymesh.auth import issue_dev_token
from querymesh.config import Settings
from querymesh.graph import compile_workflow
from querymesh.llm.fake import FakeLLM
from querymesh.observability.tracing import NoOpTracer
from querymesh.retrieval.models import RetrievedDocument

_TEST_SECRET = "integration-test-secret-key-at-least-32-bytes"


class _FakeRetrievalService:
    async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "attention text"},
                score=5.0,
                rank=1,
                retrieval_method="hybrid_rrf",
            )
        ]


def _auth_settings(**overrides) -> Settings:
    defaults = dict(JWT_SECRET_KEY=_TEST_SECRET)
    defaults.update(overrides)
    return Settings(**defaults)


def _make_client() -> TestClient:
    llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}))
    compiled_graph = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
    state = AppState(compiled_graph=compiled_graph, tracer=NoOpTracer(), cache=None)
    app = create_app(app_state=state)
    return TestClient(app, raise_server_exceptions=False)


def _valid_token(**kwargs) -> str:
    kwargs.setdefault("scopes", ["query:submit"])
    return issue_dev_token("test-user", settings=_auth_settings(), **kwargs)


@pytest.fixture(autouse=True)
def _jwt_env(monkeypatch):
    """Point Settings at a fixed JWT secret for every test in this file,
    since `require_scopes`'s dependency constructs a fresh `JWTValidator()`
    (reading global Settings) per request rather than taking one injected.
    """
    monkeypatch.setenv("JWT_SECRET_KEY", _TEST_SECRET)
    monkeypatch.delenv("JWT_PUBLIC_KEY", raising=False)
    from querymesh.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- Health stays open ---------------------------------------------------


@pytest.mark.integration
def test_health_requires_no_token() -> None:
    with _make_client() as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200


@pytest.mark.integration
def test_ready_requires_no_token() -> None:
    with _make_client() as client:
        response = client.get("/api/v1/ready")
    assert response.status_code == 200


# --- Missing token -----------------------------------------------------


@pytest.mark.integration
def test_query_without_token_is_401() -> None:
    with _make_client() as client:
        response = client.post("/api/v1/query", json={"query": "What is attention?"})

    assert response.status_code == 401
    body = response.json()
    assert body["error"] == "AuthenticationError"
    assert body["retryable"] is False


@pytest.mark.integration
def test_stream_without_token_is_401() -> None:
    with _make_client() as client:
        response = client.post("/api/v1/query/stream", json={"query": "What is attention?"})

    assert response.status_code == 401


# --- Malformed token -----------------------------------------------------


@pytest.mark.integration
def test_query_with_malformed_token_is_401() -> None:
    with _make_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "What is attention?"},
            headers={"Authorization": "Bearer not-a-real-jwt"},
        )

    assert response.status_code == 401
    assert response.json()["error"] == "AuthenticationError"


# --- Expired token -----------------------------------------------------


@pytest.mark.integration
def test_query_with_expired_token_is_401() -> None:
    now = int(time.time())
    expired = pyjwt.encode(
        {"sub": "user", "scope": "query:submit", "exp": now - 3600},
        key=_TEST_SECRET,
        algorithm="HS256",
    )
    with _make_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "What is attention?"},
            headers={"Authorization": f"Bearer {expired}"},
        )

    assert response.status_code == 401
    assert "expired" in response.json()["message"].lower()


# --- Invalid signature -----------------------------------------------------


@pytest.mark.integration
def test_query_with_wrong_signature_is_401() -> None:
    now = int(time.time())
    forged = pyjwt.encode(
        {"sub": "user", "scope": "query:submit", "exp": now + 3600},
        key="a-completely-different-secret-key-value",
        algorithm="HS256",
    )
    with _make_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "What is attention?"},
            headers={"Authorization": f"Bearer {forged}"},
        )

    assert response.status_code == 401


# --- Valid token ---------------------------------------------------------


@pytest.mark.integration
def test_query_with_valid_token_succeeds() -> None:
    token = _valid_token()
    with _make_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "What is attention?"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert len(body["citations"]) == 1


@pytest.mark.integration
def test_stream_with_valid_token_succeeds() -> None:
    token = _valid_token()
    with _make_client() as client:
        with client.stream(
            "POST",
            "/api/v1/query/stream",
            json={"query": "What is attention?"},
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line]

    assert len(lines) > 0
    first_event = json.loads(lines[0].removeprefix("data: "))
    assert first_event["type"] == "workflow_started"


# --- Insufficient permissions (valid identity, wrong scope) ----------------


@pytest.mark.integration
def test_query_with_valid_token_but_wrong_scope_is_403() -> None:
    token = _valid_token(scopes=["read:only"])  # missing query:submit
    with _make_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "What is attention?"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 403
    body = response.json()
    assert body["error"] == "AuthorizationError"
    assert body["retryable"] is False


@pytest.mark.integration
def test_query_with_no_scopes_at_all_is_403() -> None:
    token = _valid_token(scopes=[])
    with _make_client() as client:
        response = client.post(
            "/api/v1/query",
            json={"query": "What is attention?"},
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 403


# --- Response still carries request/correlation IDs on auth failure -------


@pytest.mark.integration
def test_auth_failure_response_still_has_request_id() -> None:
    with _make_client() as client:
        response = client.post("/api/v1/query", json={"query": "What is attention?"})

    body = response.json()
    assert body["request_id"]
    assert response.headers["x-request-id"] == body["request_id"]
