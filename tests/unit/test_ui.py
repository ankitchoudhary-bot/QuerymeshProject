"""Phase 21 tests: run_query_streaming (UI-agnostic, no gradio import) and
build_app (constructs a real gr.Blocks without launching a server).
"""

from __future__ import annotations

import json

import pytest

from querymesh.graph import compile_workflow
from querymesh.llm.fake import FakeLLM
from querymesh.retrieval.models import RetrievedDocument
from querymesh.ui.handlers import run_query_streaming


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


# --- run_query_streaming --------------------------------------------------


@pytest.mark.unit
async def test_streaming_yields_status_updates_in_expected_order() -> None:
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())

    statuses = []
    async for update in run_query_streaming(compiled, "what is attention?", max_retries=1):
        statuses.append(update.status)

    assert statuses[0] == "Starting research request"
    assert "Planning research..." in statuses
    assert "Searching..." in statuses
    assert "Evaluating evidence..." in statuses
    assert "Synthesizing..." in statuses
    assert "Verifying citations..." in statuses
    assert statuses[-1] == "Done"


@pytest.mark.unit
async def test_streaming_final_update_has_done_true_and_answer() -> None:
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())

    updates = [u async for u in run_query_streaming(compiled, "question", max_retries=1)]
    final = updates[-1]

    assert final.done is True
    assert final.answer != ""
    assert len(final.citations) == 1
    assert final.citations[0]["chunk_id"] == "a:chunk:0000"


@pytest.mark.unit
async def test_streaming_intermediate_updates_have_done_false() -> None:
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())

    updates = [u async for u in run_query_streaming(compiled, "question", max_retries=1)]

    assert all(u.done is False for u in updates[:-1])


@pytest.mark.unit
async def test_streaming_includes_retrieved_papers_in_final_update() -> None:
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())

    updates = [u async for u in run_query_streaming(compiled, "question", max_retries=1)]
    final = updates[-1]

    assert len(final.retrieved_papers) == 1
    assert final.retrieved_papers[0]["chunk_id"] == "a:chunk:0000"


@pytest.mark.unit
async def test_streaming_with_no_evidence_still_completes() -> None:
    llm = FakeLLM(response_text="no evidence found")
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService(docs=[]))

    updates = [u async for u in run_query_streaming(compiled, "unanswerable question", max_retries=1)]
    final = updates[-1]

    assert final.done is True
    assert final.citations == []
    assert final.retrieved_papers == []


@pytest.mark.unit
async def test_answer_accumulates_incrementally_across_multiple_updates() -> None:
    """The headline capability of this module's Phase 22 upgrade: the
    answer must arrive as a growing sequence of partial values (one per
    LLM token), not as a single jump from "" straight to the final text.
    A regression back to Phase 21's node-level-only streaming would make
    this fail (only one non-empty answer value, right at the end).
    """
    # A parseable grade response keeps grading on a single pass (no
    # retry -> no second synthesis pass re-streaming the same text).
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())

    updates = [u async for u in run_query_streaming(compiled, "question", max_retries=1)]
    distinct_nonempty_lengths = sorted({len(u.answer) for u in updates if u.answer})
    final_length = len(updates[-1].answer)

    # Multiple distinct, strictly increasing partial lengths -> genuine
    # incremental delivery, not one-shot -- and the run's own final
    # length (not a hardcoded string) is the largest value seen.
    assert len(distinct_nonempty_lengths) > 1
    assert distinct_nonempty_lengths == sorted(distinct_nonempty_lengths)
    assert distinct_nonempty_lengths[-1] == final_length


@pytest.mark.unit
async def test_citations_appear_one_at_a_time_not_all_at_once() -> None:
    """Same principle for citations: with 2 evidence docs, the citation
    count across updates should pass through 1 before reaching 2 — never
    jumping straight from 0 to 2 in a single update.
    """
    docs = [
        RetrievedDocument(
            document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "text a"},
            score=5.0,
            rank=1,
            retrieval_method="hybrid_rrf",
        ),
        RetrievedDocument(
            document={"chunk_id": "b:chunk:0001", "arxiv_id": "p2", "title": "T2", "text": "text b"},
            score=4.0,
            rank=2,
            retrieval_method="hybrid_rrf",
        ),
    ]
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService(docs=docs))

    updates = [u async for u in run_query_streaming(compiled, "question", max_retries=1)]
    citation_counts_seen = [len(u.citations) for u in updates]

    assert 1 in citation_counts_seen  # passed through exactly-one on the way to two
    assert citation_counts_seen[-1] == 2
    # Monotonically non-decreasing -- citations are only ever added, never
    # replaced or removed mid-stream.
    assert citation_counts_seen == sorted(citation_counts_seen)


# --- build_app (Gradio widget wiring) --------------------------------------


@pytest.mark.unit
def test_build_app_returns_blocks_instance() -> None:
    import gradio as gr

    from querymesh.ui import build_app

    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())

    app = build_app(compiled)

    assert isinstance(app, gr.Blocks)


@pytest.mark.unit
def test_build_app_contains_expected_components() -> None:
    from querymesh.ui import build_app

    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())

    app = build_app(compiled)
    component_types = {type(c).__name__ for c in app.blocks.values()}

    assert "Textbox" in component_types
    assert "Button" in component_types
    assert "JSON" in component_types
    assert "Markdown" in component_types
