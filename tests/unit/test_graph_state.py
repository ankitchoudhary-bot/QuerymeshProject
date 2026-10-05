"""Phase 12 tests: ResearchState creation, field defaults, and
JSON-serializability via to_serializable_dict.
"""

from __future__ import annotations

import json

import pytest

from querymesh.agents.models import Citation, DocumentGradeResult, GradedDocument
from querymesh.graph.state import create_initial_state, to_serializable_dict
from querymesh.retrieval.models import RetrievedDocument


def _rd(chunk_id: str) -> RetrievedDocument:
    return RetrievedDocument(
        document={"chunk_id": chunk_id, "arxiv_id": "p1", "title": "T", "text": "t"},
        score=1.0,
        rank=1,
        retrieval_method="hybrid_rrf",
    )


@pytest.mark.unit
def test_create_initial_state_defaults() -> None:
    state = create_initial_state("what is attention?")

    assert state["query"] == "what is attention?"
    assert state["search_queries"] == []
    assert state["retrieved_documents"] == []
    assert state["graded_documents"] == []
    assert state["evidence"] == []
    assert state["answer"] == ""
    assert state["citations"] == []
    assert state["retry_count"] == 0
    assert state["max_retries"] == 2
    assert state["errors"] == []
    assert state["metadata"] == {}


@pytest.mark.unit
def test_create_initial_state_custom_max_retries() -> None:
    state = create_initial_state("query", max_retries=5)
    assert state["max_retries"] == 5


@pytest.mark.unit
def test_state_fields_are_mutable_and_independent_across_instances() -> None:
    state1 = create_initial_state("q1")
    state2 = create_initial_state("q2")

    state1["search_queries"].append("extra query")

    assert state1["search_queries"] == ["extra query"]
    assert state2["search_queries"] == []  # not shared/mutated


@pytest.mark.unit
def test_to_serializable_dict_is_json_dumpable() -> None:
    state = create_initial_state("what is attention?")
    state["retrieved_documents"] = [_rd("a:chunk:0000")]
    state["graded_documents"] = [
        GradedDocument(
            chunk_id="a:chunk:0000",
            document={"text": "t"},
            grade=DocumentGradeResult(relevant=True, score=0.9, reason="good"),
        )
    ]
    state["citations"] = [Citation(arxiv_id="p1", title="T", chunk_id="a:chunk:0000", claim="claim")]
    state["answer"] = "the answer"
    state["errors"] = ["one warning"]

    serializable = to_serializable_dict(state)
    dumped = json.dumps(serializable)  # must not raise
    reloaded = json.loads(dumped)

    assert reloaded["query"] == "what is attention?"
    assert reloaded["answer"] == "the answer"
    assert reloaded["retrieved_documents"][0]["document"]["chunk_id"] == "a:chunk:0000"
    assert reloaded["graded_documents"][0]["grade"]["score"] == 0.9
    assert reloaded["citations"][0]["chunk_id"] == "a:chunk:0000"
    assert reloaded["errors"] == ["one warning"]


@pytest.mark.unit
def test_to_serializable_dict_handles_empty_state() -> None:
    state = create_initial_state("query")
    serializable = to_serializable_dict(state)
    json.dumps(serializable)  # must not raise

    assert serializable["retrieved_documents"] == []
    assert serializable["citations"] == []
