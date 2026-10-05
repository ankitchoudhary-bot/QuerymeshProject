"""Phase 29 tests: input validation limits (max query length, max
document count), MCP static bearer-token auth, and graph recursion limit
wiring.
"""

from __future__ import annotations

import json

import pytest

from querymesh.errors import ValidationError
from querymesh.graph import compile_workflow, create_initial_state, run_research_workflow
from querymesh.llm.fake import FakeLLM
from querymesh.retrieval.models import RetrievedDocument
from querymesh.retrieval.service import RetrievalService
from querymesh.security.limits import MAX_QUERY_LENGTH, MAX_TOP_K, validate_query, validate_top_k
from querymesh.security.mcp_auth import StaticBearerTokenVerifier


# --- validate_query --------------------------------------------------------


@pytest.mark.unit
def test_validate_query_accepts_normal_query() -> None:
    assert validate_query("What is attention?") == "What is attention?"


@pytest.mark.unit
def test_validate_query_strips_whitespace() -> None:
    assert validate_query("  hello  ") == "hello"


@pytest.mark.unit
def test_validate_query_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        validate_query("")


@pytest.mark.unit
def test_validate_query_rejects_whitespace_only() -> None:
    with pytest.raises(ValidationError):
        validate_query("   ")


@pytest.mark.unit
def test_validate_query_rejects_none() -> None:
    with pytest.raises(ValidationError):
        validate_query(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_validate_query_rejects_too_long() -> None:
    with pytest.raises(ValidationError):
        validate_query("x" * (MAX_QUERY_LENGTH + 1))


@pytest.mark.unit
def test_validate_query_accepts_exactly_max_length() -> None:
    query = "x" * MAX_QUERY_LENGTH
    assert validate_query(query) == query


# --- validate_top_k --------------------------------------------------------


@pytest.mark.unit
def test_validate_top_k_accepts_normal_value() -> None:
    assert validate_top_k(10) == 10


@pytest.mark.unit
def test_validate_top_k_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        validate_top_k(0)


@pytest.mark.unit
def test_validate_top_k_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        validate_top_k(-5)


@pytest.mark.unit
def test_validate_top_k_rejects_too_large() -> None:
    with pytest.raises(ValidationError):
        validate_top_k(MAX_TOP_K + 1)


@pytest.mark.unit
def test_validate_top_k_accepts_exactly_max() -> None:
    assert validate_top_k(MAX_TOP_K) == MAX_TOP_K


# --- create_initial_state validates the query ------------------------------


@pytest.mark.unit
def test_create_initial_state_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        create_initial_state("")


@pytest.mark.unit
def test_create_initial_state_rejects_too_long_query() -> None:
    with pytest.raises(ValidationError):
        create_initial_state("x" * (MAX_QUERY_LENGTH + 1))


@pytest.mark.unit
def test_create_initial_state_strips_and_accepts_valid_query() -> None:
    state = create_initial_state("  a valid query  ")
    assert state["query"] == "a valid query"


# --- RetrievalService validates query and top_k ----------------------------


class _FakeHybridRetriever:
    async def search(self, query, top_k=10, filters=None) -> list[RetrievedDocument]:
        return []


@pytest.mark.unit
async def test_retrieval_service_rejects_empty_query() -> None:
    service = RetrievalService(hybrid_retriever=_FakeHybridRetriever())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        await service.retrieve("")


@pytest.mark.unit
async def test_retrieval_service_rejects_excessive_top_k() -> None:
    service = RetrievalService(hybrid_retriever=_FakeHybridRetriever())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        await service.retrieve("valid query", top_k=MAX_TOP_K + 1)


@pytest.mark.unit
async def test_retrieval_service_accepts_valid_input() -> None:
    service = RetrievalService(hybrid_retriever=_FakeHybridRetriever())  # type: ignore[arg-type]
    results = await service.retrieve("valid query", top_k=10)
    assert results == []


# --- MCP static bearer-token auth ------------------------------------------


@pytest.mark.unit
def test_static_bearer_token_verifier_requires_nonempty_token() -> None:
    with pytest.raises(ValueError):
        StaticBearerTokenVerifier("")


@pytest.mark.unit
async def test_static_bearer_token_verifier_accepts_correct_token() -> None:
    verifier = StaticBearerTokenVerifier("secret-token")
    result = await verifier.verify_token("secret-token")
    assert result is not None
    assert result.token == "secret-token"


@pytest.mark.unit
async def test_static_bearer_token_verifier_rejects_wrong_token() -> None:
    verifier = StaticBearerTokenVerifier("secret-token")
    result = await verifier.verify_token("wrong-token")
    assert result is None


@pytest.mark.unit
async def test_static_bearer_token_verifier_rejects_empty_token() -> None:
    verifier = StaticBearerTokenVerifier("secret-token")
    result = await verifier.verify_token("")
    assert result is None


@pytest.mark.unit
def test_mcp_server_with_auth_wires_verifier_through() -> None:
    """FastMCP's in-memory test transport doesn't carry auth headers (only
    real HTTP/SSE transports do — verified by attempting it: it raises
    "This transport does not support auth"), so the end-to-end
    accept/reject behavior is proven at the verifier level (tests above,
    directly against `verify_token`). This test confirms the *wiring*:
    the server actually receives and holds the auth provider it's given.
    """
    from querymesh.agents.researcher import ResearcherAgent
    from querymesh.mcp.servers import build_research_server

    llm = FakeLLM(response_text=json.dumps({"search_queries": ["q"]}))
    verifier = StaticBearerTokenVerifier("correct-token")
    server = build_research_server(ResearcherAgent(llm), auth=verifier)

    assert server.auth is verifier


@pytest.mark.unit
async def test_mcp_server_without_auth_still_works_unauthenticated() -> None:
    """Default (no auth given) behavior is unchanged — servers stay
    usable for trusted-network deployments without requiring a token.
    """
    from fastmcp import Client

    from querymesh.agents.researcher import ResearcherAgent
    from querymesh.mcp.servers import build_research_server

    llm = FakeLLM(response_text=json.dumps({"search_queries": ["q1"]}))
    server = build_research_server(ResearcherAgent(llm))  # no auth

    async with Client(server) as client:
        result = await client.call_tool("search_papers", {"query": "q"})

    assert result.data["search_queries"] == ["q1"]


# --- Graph recursion limit --------------------------------------------------


@pytest.mark.unit
async def test_run_research_workflow_passes_recursion_limit() -> None:
    """A real run still completes normally with the recursion_limit wired
    through — this isn't just a config no-op.
    """
    llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}))

    class _FakeRetrievalService:
        async def retrieve(self, query, top_k=10, filters=None):
            return [
                RetrievedDocument(
                    document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "t"},
                    score=1.0,
                    rank=1,
                    retrieval_method="hybrid_rrf",
                )
            ]

    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())  # type: ignore[arg-type]
    state = create_initial_state("a valid query", max_retries=1)

    result = await run_research_workflow(compiled, state, recursion_limit=50)

    assert "answer" in result
