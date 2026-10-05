# """Phase 22 tests: SynthesizerAgent.synthesize_streaming (token-level),
# StreamEvent construction, and run_research_workflow_streaming end to end.
# """

# from __future__ import annotations

# import json

# import pytest

# from querymesh.agents.synthesizer import SynthesizerAgent
# from querymesh.graph import (
#     AGENT_STARTED,
#     CITATION,
#     GENERATION_STARTED,
#     GRADING_COMPLETED,
#     RETRIEVAL_COMPLETED,
#     RETRIEVAL_STARTED,
#     TOKEN,
#     WORKFLOW_COMPLETED,
#     WORKFLOW_STARTED,
#     compile_workflow,
#     create_initial_state,
#     make_event,
#     run_research_workflow_streaming,
# )
# from querymesh.llm.base import LLMChunk, LLMResponse
# from querymesh.llm.fake import FakeLLM
# from querymesh.retrieval.models import RetrievedDocument


# def _rd(chunk_id: str) -> RetrievedDocument:
#     return RetrievedDocument(
#         document={"chunk_id": chunk_id, "arxiv_id": "p1", "title": "T", "text": "t"},
#         score=1.0,
#         rank=1,
#         retrieval_method="hybrid_rrf",
#     )


# class _FakeRetrievalService:
#     def __init__(self, docs: list[RetrievedDocument] | None = None) -> None:
#         self.docs = docs if docs is not None else [_rd("a:chunk:0000")]

#     async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
#         return self.docs


# def _grade_json(relevant: bool = True, score: float = 0.9) -> str:
#     return json.dumps({"relevant": relevant, "score": score, "reason": "test"})


# # --- make_event -----------------------------------------------------------


# @pytest.mark.unit
# def test_make_event_has_all_typed_dict_fields() -> None:
#     event = make_event("token", "hello", agent="synthesizer", extra=1)

#     assert event["type"] == "token"
#     assert event["message"] == "hello"
#     assert event["agent"] == "synthesizer"
#     assert event["data"] == {"extra": 1}


# @pytest.mark.unit
# def test_make_event_defaults() -> None:
#     event = make_event("workflow_started")
#     assert event["agent"] is None
#     assert event["message"] == ""
#     assert event["data"] == {}


# # --- SynthesizerAgent.synthesize_streaming ---------------------------------


# @pytest.mark.unit
# async def test_synthesize_streaming_yields_chunks_then_result() -> None:
#     llm = FakeLLM(response_text="one two three")
#     agent = SynthesizerAgent(llm)

#     items = [item async for item in agent.synthesize_streaming("q", [_rd("a:chunk:0000")])]

#     chunks = items[:-1]
#     result = items[-1]

#     assert all(isinstance(c, LLMChunk) for c in chunks)
#     assert not hasattr(result, "text")  # it's a SynthesisResult, not an LLMChunk
#     assert result.answer == "one two three"
#     assert result.evidence_sufficient is True


# @pytest.mark.unit
# async def test_synthesize_streaming_concatenates_chunks_correctly() -> None:
#     llm = FakeLLM(response_text="alpha beta gamma")
#     agent = SynthesizerAgent(llm)

#     items = [item async for item in agent.synthesize_streaming("q", [])]
#     result = items[-1]

#     assert result.answer == "alpha beta gamma"
#     assert result.evidence_sufficient is False  # empty evidence


# @pytest.mark.unit
# async def test_synthesize_streaming_uses_real_astream() -> None:
#     """Verify it actually streams (multiple chunks), not just wraps ainvoke."""

#     class _MultiChunkLLM:
#         model = "multi"

#         async def ainvoke(self, prompt: str) -> LLMResponse:
#             raise AssertionError("should not call ainvoke in streaming path")

#         async def astream(self, prompt: str):
#             for word in ["hello", " world"]:
#                 yield LLMChunk(text=word, model=self.model)

#     agent = SynthesizerAgent(_MultiChunkLLM())  # type: ignore[arg-type]
#     items = [item async for item in agent.synthesize_streaming("q", [_rd("a:chunk:0000")])]

#     assert len(items) == 3  # 2 chunks + final result
#     assert items[-1].answer == "hello world"


# # --- run_research_workflow_streaming ---------------------------------------


# @pytest.mark.unit
# async def test_streaming_workflow_starts_and_completes() -> None:
#     llm = FakeLLM(response_text=_grade_json())
#     compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
#     state = create_initial_state("q", max_retries=1)

#     events = [e async for e in run_research_workflow_streaming(compiled, state)]

#     assert events[0]["type"] == WORKFLOW_STARTED
#     assert events[-1]["type"] == WORKFLOW_COMPLETED


# @pytest.mark.unit
# async def test_streaming_workflow_emits_all_spec_event_types() -> None:
#     llm = FakeLLM(response_text=_grade_json())
#     compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
#     state = create_initial_state("q", max_retries=1)

#     events = [e async for e in run_research_workflow_streaming(compiled, state)]
#     types_seen = {e["type"] for e in events}

#     assert AGENT_STARTED in types_seen
#     assert RETRIEVAL_STARTED in types_seen
#     assert RETRIEVAL_COMPLETED in types_seen
#     assert GRADING_COMPLETED in types_seen
#     assert GENERATION_STARTED in types_seen
#     assert TOKEN in types_seen
#     assert CITATION in types_seen


# @pytest.mark.unit
# async def test_streaming_workflow_token_events_reconstruct_answer() -> None:
#     llm = FakeLLM(response_text="the reconstructed answer")
#     compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
#     # max_retries=0: grading this non-JSON response fails to parse (defaults
#     # to not-relevant), which would otherwise trigger a second synthesis
#     # pass on retry -- pin to a single pass so token events map 1:1 to the
#     # single final answer.
#     state = create_initial_state("q", max_retries=0)

#     events = [e async for e in run_research_workflow_streaming(compiled, state)]
#     token_events = [e for e in events if e["type"] == TOKEN]
#     reconstructed = "".join(e["message"] for e in token_events)

#     completed = next(e for e in events if e["type"] == WORKFLOW_COMPLETED)
#     assert reconstructed == completed["data"]["answer"]


# @pytest.mark.unit
# async def test_streaming_workflow_completed_event_has_citation_count() -> None:
#     llm = FakeLLM(response_text=_grade_json())
#     compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
#     state = create_initial_state("q", max_retries=1)

#     events = [e async for e in run_research_workflow_streaming(compiled, state)]
#     completed = next(e for e in events if e["type"] == WORKFLOW_COMPLETED)

#     assert completed["data"]["citation_count"] == 1


# @pytest.mark.unit
# async def test_streaming_workflow_with_no_evidence_still_completes_cleanly() -> None:
#     llm = FakeLLM(response_text="no evidence")
#     compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService(docs=[]))
#     state = create_initial_state("unanswerable", max_retries=1)

#     events = [e async for e in run_research_workflow_streaming(compiled, state)]

#     assert events[-1]["type"] == WORKFLOW_COMPLETED
#     assert events[-1]["data"]["citation_count"] == 0




"""Phase 22 tests: SynthesizerAgent.synthesize_streaming (token-level),
StreamEvent construction, and run_research_workflow_streaming end to end.
"""

from __future__ import annotations

import json

import pytest

from querymesh.agents.synthesizer import SynthesizerAgent
from querymesh.graph import (
    AGENT_STARTED,
    CITATION,
    GENERATION_STARTED,
    GRADING_COMPLETED,
    RETRIEVAL_COMPLETED,
    RETRIEVAL_STARTED,
    TOKEN,
    WORKFLOW_COMPLETED,
    WORKFLOW_STARTED,
    compile_workflow,
    create_initial_state,
    make_event,
    run_research_workflow_streaming,
)
from querymesh.llm.base import LLMChunk, LLMResponse
from querymesh.llm.fake import FakeLLM
from querymesh.retrieval.models import RetrievedDocument


def _rd(chunk_id: str) -> RetrievedDocument:
    return RetrievedDocument(
        document={"chunk_id": chunk_id, "arxiv_id": "p1", "title": "T", "text": "t"},
        score=1.0,
        rank=1,
        retrieval_method="hybrid_rrf",
    )


class _FakeRetrievalService:
    def __init__(self, docs: list[RetrievedDocument] | None = None) -> None:
        self.docs = docs if docs is not None else [_rd("a:chunk:0000")]

    async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        return self.docs


def _grade_json(relevant: bool = True, score: float = 0.9) -> str:
    return json.dumps({"relevant": relevant, "score": score, "reason": "test"})


# --- make_event -----------------------------------------------------------


@pytest.mark.unit
def test_make_event_has_all_typed_dict_fields() -> None:
    event = make_event("token", "hello", agent="synthesizer", extra=1)

    assert event["type"] == "token"
    assert event["message"] == "hello"
    assert event["agent"] == "synthesizer"
    assert event["data"] == {"extra": 1}


@pytest.mark.unit
def test_make_event_defaults() -> None:
    event = make_event("workflow_started")
    assert event["agent"] is None
    assert event["message"] == ""
    assert event["data"] == {}


# --- SynthesizerAgent.synthesize_streaming ---------------------------------


@pytest.mark.unit
async def test_synthesize_streaming_yields_chunks_then_result() -> None:
    llm = FakeLLM(response_text="one two three")
    agent = SynthesizerAgent(llm)

    items = [item async for item in agent.synthesize_streaming("q", [_rd("a:chunk:0000")])]

    chunks = items[:-1]
    result = items[-1]

    assert all(isinstance(c, LLMChunk) for c in chunks)
    assert not hasattr(result, "text")  # it's a SynthesisResult, not an LLMChunk
    assert result.answer == "one two three"
    assert result.evidence_sufficient is True


@pytest.mark.unit
async def test_synthesize_streaming_concatenates_chunks_correctly() -> None:
    llm = FakeLLM(response_text="alpha beta gamma")
    agent = SynthesizerAgent(llm)

    items = [item async for item in agent.synthesize_streaming("q", [])]
    result = items[-1]

    assert result.answer == "alpha beta gamma"
    assert result.evidence_sufficient is False  # empty evidence


@pytest.mark.unit
async def test_synthesize_streaming_uses_real_astream() -> None:
    """Verify it actually streams (multiple chunks), not just wraps ainvoke."""

    class _MultiChunkLLM:
        model = "multi"

        async def ainvoke(self, prompt: str) -> LLMResponse:
            raise AssertionError("should not call ainvoke in streaming path")

        async def astream(self, prompt: str):
            for word in ["hello", " world"]:
                yield LLMChunk(text=word, model=self.model)

    agent = SynthesizerAgent(_MultiChunkLLM())  # type: ignore[arg-type]
    items = [item async for item in agent.synthesize_streaming("q", [_rd("a:chunk:0000")])]

    assert len(items) == 3  # 2 chunks + final result
    assert items[-1].answer == "hello world"


# --- run_research_workflow_streaming ---------------------------------------


@pytest.mark.unit
async def test_streaming_workflow_starts_and_completes() -> None:
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
    state = create_initial_state("q", max_retries=1)

    events = [e async for e in run_research_workflow_streaming(compiled, state)]

    assert events[0]["type"] == WORKFLOW_STARTED
    assert events[-1]["type"] == WORKFLOW_COMPLETED


@pytest.mark.unit
async def test_streaming_workflow_emits_all_spec_event_types() -> None:
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
    state = create_initial_state("q", max_retries=1)

    events = [e async for e in run_research_workflow_streaming(compiled, state)]
    types_seen = {e["type"] for e in events}

    assert AGENT_STARTED in types_seen
    assert RETRIEVAL_STARTED in types_seen
    assert RETRIEVAL_COMPLETED in types_seen
    assert GRADING_COMPLETED in types_seen
    assert GENERATION_STARTED in types_seen
    assert TOKEN in types_seen
    assert CITATION in types_seen


@pytest.mark.unit
async def test_streaming_workflow_token_events_reconstruct_answer() -> None:
    llm = FakeLLM(response_text="the reconstructed answer")
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
    # max_retries=0: grading this non-JSON response fails to parse (defaults
    # to not-relevant), which would otherwise trigger a second synthesis
    # pass on retry -- pin to a single pass so token events map 1:1 to the
    # single final answer.
    state = create_initial_state("q", max_retries=0)

    events = [e async for e in run_research_workflow_streaming(compiled, state)]
    token_events = [e for e in events if e["type"] == TOKEN]
    reconstructed = "".join(e["message"] for e in token_events)

    completed = next(e for e in events if e["type"] == WORKFLOW_COMPLETED)
    assert reconstructed == completed["data"]["answer"]


@pytest.mark.unit
async def test_streaming_workflow_completed_event_has_citation_count() -> None:
    llm = FakeLLM(response_text=_grade_json())
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService())
    state = create_initial_state("q", max_retries=1)

    events = [e async for e in run_research_workflow_streaming(compiled, state)]
    completed = next(e for e in events if e["type"] == WORKFLOW_COMPLETED)

    assert completed["data"]["citation_count"] == 1


@pytest.mark.unit
async def test_streaming_workflow_with_no_evidence_still_completes_cleanly() -> None:
    llm = FakeLLM(response_text="no evidence")
    compiled = compile_workflow(llm=llm, retrieval_service=_FakeRetrievalService(docs=[]))
    state = create_initial_state("unanswerable", max_retries=1)

    events = [e async for e in run_research_workflow_streaming(compiled, state)]

    assert events[-1]["type"] == WORKFLOW_COMPLETED
    assert events[-1]["data"]["citation_count"] == 0
