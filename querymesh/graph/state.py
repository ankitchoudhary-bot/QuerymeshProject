"""Shared graph state for QueryMesh's research workflow.

`ResearchState` is a `TypedDict` — LangGraph (Phase 13) passes this dict
between nodes. It holds richer objects (`RetrievedDocument`, `Citation`,
etc. — themselves Pydantic models) rather than raw dicts, for type safety
within the graph; `to_serializable_dict()` below converts the whole state
to plain JSON-compatible data for logging/checkpointing/observability,
per the rule "keep state serializable wherever practical."
"""

from __future__ import annotations

from typing import Any, TypedDict

from querymesh.agents.models import Citation, GradedDocument
from querymesh.retrieval.models import RetrievedDocument
from querymesh.security.limits import validate_query

_DEFAULT_MAX_RETRIES = 2


class ResearchState(TypedDict):
    query: str
    search_queries: list[str]
    retrieved_documents: list[RetrievedDocument]
    graded_documents: list[GradedDocument]
    evidence: list[RetrievedDocument]
    answer: str
    citations: list[Citation]
    retry_count: int
    max_retries: int
    errors: list[str]
    metadata: dict[str, Any]
    grading_reasons: list[str]


def create_initial_state(query: str, max_retries: int = _DEFAULT_MAX_RETRIES) -> ResearchState:
    """Build a fresh `ResearchState` for a new user query.

    `query` is validated here (max length, non-empty) — this is the single
    true entry point for a user's question into the graph, so it's the
    right place to enforce that production safety limit once, rather than
    scattered across every node that touches `state["query"]`.
    """
    return ResearchState(
        query=validate_query(query),
        search_queries=[],
        retrieved_documents=[],
        graded_documents=[],
        evidence=[],
        answer="",
        citations=[],
        retry_count=0,
        max_retries=max_retries,
        errors=[],
        metadata={},
        grading_reasons=[]
    )


def to_serializable_dict(state: ResearchState) -> dict[str, Any]:
    """Convert a `ResearchState` to plain JSON-compatible data.

    Pydantic model fields (`retrieved_documents`, `graded_documents`,
    `evidence`, `citations`) are converted via `.model_dump()`; everything
    else is already JSON-safe.
    """

    def _dump_list(items: list) -> list:
        return [item.model_dump() if hasattr(item, "model_dump") else item for item in items]

    return {
        "query": state["query"],
        "search_queries": list(state["search_queries"]),
        "retrieved_documents": _dump_list(state["retrieved_documents"]),
        "graded_documents": _dump_list(state["graded_documents"]),
        "evidence": _dump_list(state["evidence"]),
        "answer": state["answer"],
        "citations": _dump_list(state["citations"]),
        "retry_count": state["retry_count"],
        "max_retries": state["max_retries"],
        "errors": list(state["errors"]),
        "metadata": dict(state["metadata"]),
        "grading_reasons": list(state["grading_reasons"])
    }
