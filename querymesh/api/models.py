"""API request/response models.

Kept separate from `querymesh.agents.models`/`querymesh.retrieval.models`
even where they overlap in shape — the API's contract with clients should
be able to evolve independently of internal graph/agent types (rule:
"do not move business logic into the API layer" implies the reverse too:
don't let the API layer's shape dictate internal types).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from querymesh.security.limits import MAX_QUERY_LENGTH, MAX_TOP_K


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    max_retries: int = Field(default=2, ge=0, le=5)


class CitationResponse(BaseModel):
    arxiv_id: str
    title: str
    chunk_id: str
    claim: str


class QueryResponse(BaseModel):
    request_id: str
    correlation_id: str
    query: str
    answer: str
    citations: list[CitationResponse]
    retrieved_document_count: int
    retry_count: int
    errors: list[str]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, bool]


class ErrorResponse(BaseModel):
    error: str
    message: str
    retryable: bool
    request_id: str
    correlation_id: str
