# """Phase 13 tests: the compiled LangGraph research graph — happy path,
# retry loop (bounded), and graceful termination when evidence never
# becomes sufficient.
# """

# from __future__ import annotations

# import json

# import pytest

# from querymesh.graph import compile_workflow, create_initial_state
# from querymesh.graph.edges import FINISH, RETRY, route_after_supervisor
# from querymesh.llm.base import LLMResponse
# from querymesh.retrieval.models import RetrievedDocument


# def _doc(chunk_id: str) -> RetrievedDocument:
#     return RetrievedDocument(
#         document={"chunk_id": chunk_id, "arxiv_id": "p1", "title": "Attention Paper", "text": "attention text"},
#         score=5.0,
#         rank=1,
#         retrieval_method="hybrid_rrf",
#     )


# class _FakeRetrievalService:
#     def __init__(self, docs: list[RetrievedDocument]) -> None:
#         self.docs = docs
#         self.call_count = 0

#     async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
#         self.call_count += 1
#         return self.docs


# class _ScriptedLLM:
#     """Returns each response in order, repeating the last one once exhausted."""

#     model = "scripted"

#     def __init__(self, responses: list[str]) -> None:
#         self._responses = responses
#         self._index = 0

#     async def ainvoke(self, prompt: str) -> LLMResponse:
#         text = self._responses[min(self._index, len(self._responses) - 1)]
#         self._index += 1
#         return LLMResponse(text=text, model=self.model)

#     async def astream(self, prompt: str):
#         from querymesh.llm.base import LLMChunk

#         text = self._responses[min(self._index, len(self._responses) - 1)]
#         self._index += 1
#         yield LLMChunk(text=text, model=self.model, is_final=True)


# def _plan_json(queries: list[str]) -> str:
#     return json.dumps({"search_queries": queries, "reasoning": "test plan"})


# def _grade_json(relevant: bool, score: float) -> str:
#     return json.dumps({"relevant": relevant, "score": score, "reason": "test grade"})


# # --- Routing function (pure, no graph needed) --------------------------


# @pytest.mark.unit
# def test_route_finishes_when_evidence_sufficient() -> None:
#     state = create_initial_state("q")
#     state["metadata"]["evidence_sufficient"] = True
#     assert route_after_supervisor(state) == FINISH


# @pytest.mark.unit
# def test_route_retries_when_evidence_insufficient_and_budget_remains() -> None:
#     state = create_initial_state("q", max_retries=2)
#     state["metadata"]["evidence_sufficient"] = False
#     state["retry_count"] = 0
#     assert route_after_supervisor(state) == RETRY


# @pytest.mark.unit
# def test_route_finishes_when_retry_budget_exhausted() -> None:
#     state = create_initial_state("q", max_retries=1)
#     state["metadata"]["evidence_sufficient"] = False
#     state["retry_count"] = 1
#     assert route_after_supervisor(state) == FINISH


# @pytest.mark.unit
# def test_route_defaults_to_finish_when_metadata_missing_flag() -> None:
#     state = create_initial_state("q")
#     assert route_after_supervisor(state) == FINISH


# # --- Compiled graph, happy path (single pass, no retry needed) -----------


# @pytest.mark.unit
# async def test_graph_happy_path_produces_answer_and_citations() -> None:
#     # research JSON, then grade JSON (llm shared between researcher+grader here)
#     llm = _ScriptedLLM([_plan_json(["attention query"]), _grade_json(True, 0.9), "Final grounded answer."])
#     retrieval_service = _FakeRetrievalService([_doc("a:chunk:0000")])

#     compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)
#     result = await compiled.ainvoke(create_initial_state("What is attention?", max_retries=2))

#     assert result["answer"] == "Final grounded answer."
#     assert len(result["citations"]) == 1
#     assert result["citations"][0].chunk_id == "a:chunk:0000"
#     assert result["retry_count"] == 0
#     assert result["errors"] == []


# @pytest.mark.unit
# async def test_graph_records_search_queries_from_researcher() -> None:
#     llm = _ScriptedLLM([_plan_json(["query a", "query b"]), _grade_json(True, 0.9), "answer"])
#     retrieval_service = _FakeRetrievalService([_doc("a:chunk:0000")])

#     compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)
#     result = await compiled.ainvoke(create_initial_state("question", max_retries=2))

#     assert result["search_queries"] == ["query a", "query b"]


# # --- rewrite_query node, in isolation --------------------------------------


# @pytest.mark.unit
# async def test_rewrite_query_node_updates_search_queries() -> None:
#     from querymesh.agents import CitationAgent, GraderAgent, RetrieverAgent, SynthesizerAgent
#     from querymesh.agents.researcher import ResearcherAgent
#     from querymesh.graph.nodes import GraphNodes

#     llm = _ScriptedLLM([_plan_json(["broadened query"])])
#     nodes = GraphNodes(
#         researcher=ResearcherAgent(llm),  # type: ignore[arg-type]
#         retriever=RetrieverAgent(_FakeRetrievalService([])),  # type: ignore[arg-type]
#         grader=GraderAgent(llm),  # type: ignore[arg-type]
#         synthesizer=SynthesizerAgent(llm),  # type: ignore[arg-type]
#         citation=CitationAgent(),
#     )

#     state = create_initial_state("original question")
#     state["search_queries"] = ["old narrow query"]

#     update = await nodes.rewrite_query(state)

#     assert update["search_queries"] == ["broadened query"]


# # --- Retry loop -------------------------------------------------------------


# @pytest.mark.unit
# async def test_graph_retries_when_grading_yields_no_evidence_then_succeeds() -> None:
#     """First pass: grader rejects everything -> evidence empty -> synthesizer
#     flags insufficient -> supervisor routes to retry -> rewrite_query produces
#     new search_queries -> second pass: grader accepts -> finishes.
#     """
#     llm = _ScriptedLLM(
#         [
#             _plan_json(["query one"]),  # research (1st pass)
#             _grade_json(False, 0.1),  # grade (1st pass) -> rejected, evidence empty
#             "no evidence available",  # synthesize (1st pass)
#             _plan_json(["query one, broadened"]),  # rewrite_query (before retry pass)
#             _grade_json(True, 0.9),  # grade (2nd pass, after retry) -> accepted
#             "Final grounded answer after retry.",  # synthesize (2nd pass)
#         ]
#     )
#     retrieval_service = _FakeRetrievalService([_doc("a:chunk:0000")])

#     compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)
#     result = await compiled.ainvoke(create_initial_state("question", max_retries=2))

#     assert result["retry_count"] == 1
#     assert result["answer"] == "Final grounded answer after retry."
#     assert len(result["citations"]) == 1
#     assert retrieval_service.call_count == 2  # retrieval ran twice
#     assert result["search_queries"] == ["query one, broadened"]  # rewritten, not resent


# @pytest.mark.unit
# async def test_graph_terminates_deterministically_when_evidence_never_sufficient() -> None:
#     """Grader always rejects everything -> retries exhaust at max_retries,
#     then the graph finishes anyway (never loops forever).
#     """
#     always_reject_llm = _ScriptedLLM([_plan_json(["q"]), _grade_json(False, 0.0), "no evidence"])
#     # Empty docs -> grading never finds anything relevant, every pass.
#     retrieval_service = _FakeRetrievalService([])

#     compiled = compile_workflow(llm=always_reject_llm, retrieval_service=retrieval_service)
#     result = await compiled.ainvoke(create_initial_state("question", max_retries=2))

#     assert result["retry_count"] == 2  # capped at max_retries
#     assert result["evidence"] == []
#     # Graph terminated (didn't hang/raise a recursion error) -- reaching
#     # this assertion at all is the real proof.


"""Phase 13 tests: the compiled LangGraph research graph — happy path,
retry loop (bounded), and graceful termination when evidence never
becomes sufficient.
"""

from __future__ import annotations

import json

import pytest

from querymesh.graph import compile_workflow, create_initial_state
from querymesh.graph.edges import FINISH, RETRY, route_after_supervisor
from querymesh.llm.base import LLMResponse
from querymesh.llm.fake import FakeLLM
from querymesh.retrieval.models import RetrievedDocument


def _doc(chunk_id: str) -> RetrievedDocument:
    return RetrievedDocument(
        document={"chunk_id": chunk_id, "arxiv_id": "p1", "title": "Attention Paper", "text": "attention text"},
        score=5.0,
        rank=1,
        retrieval_method="hybrid_rrf",
    )


class _FakeRetrievalService:
    def __init__(self, docs: list[RetrievedDocument]) -> None:
        self.docs = docs
        self.call_count = 0

    async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        self.call_count += 1
        return self.docs


class _ScriptedLLM:
    """Returns each response in order, repeating the last one once exhausted."""

    model = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    async def ainvoke(self, prompt: str) -> LLMResponse:
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return LLMResponse(text=text, model=self.model)

    async def astream(self, prompt: str):
        from querymesh.llm.base import LLMChunk

        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        yield LLMChunk(text=text, model=self.model, is_final=True)


def _plan_json(queries: list[str]) -> str:
    return json.dumps({"search_queries": queries, "reasoning": "test plan"})


def _grade_json(relevant: bool, score: float) -> str:
    return json.dumps({"relevant": relevant, "score": score, "reason": "test grade"})


# --- Routing function (pure, no graph needed) --------------------------


@pytest.mark.unit
def test_route_finishes_when_evidence_sufficient() -> None:
    state = create_initial_state("q")
    state["metadata"]["evidence_sufficient"] = True
    assert route_after_supervisor(state) == FINISH


@pytest.mark.unit
def test_route_retries_when_evidence_insufficient_and_budget_remains() -> None:
    state = create_initial_state("q", max_retries=2)
    state["metadata"]["evidence_sufficient"] = False
    state["retry_count"] = 0
    assert route_after_supervisor(state) == RETRY


@pytest.mark.unit
def test_route_finishes_when_retry_budget_exhausted() -> None:
    state = create_initial_state("q", max_retries=1)
    state["metadata"]["evidence_sufficient"] = False
    state["retry_count"] = 1
    assert route_after_supervisor(state) == FINISH


@pytest.mark.unit
def test_route_defaults_to_finish_when_metadata_missing_flag() -> None:
    state = create_initial_state("q")
    assert route_after_supervisor(state) == FINISH


# --- Compiled graph, happy path (single pass, no retry needed) -----------


@pytest.mark.unit
async def test_graph_happy_path_produces_answer_and_citations() -> None:
    # research JSON, then grade JSON (llm shared between researcher+grader here)
    llm = _ScriptedLLM([_plan_json(["attention query"]), _grade_json(True, 0.9), "Final grounded answer."])
    retrieval_service = _FakeRetrievalService([_doc("a:chunk:0000")])

    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)
    result = await compiled.ainvoke(create_initial_state("What is attention?", max_retries=2))

    assert result["answer"] == "Final grounded answer."
    assert len(result["citations"]) == 1
    assert result["citations"][0].chunk_id == "a:chunk:0000"
    assert result["retry_count"] == 0
    assert result["errors"] == []


@pytest.mark.unit
async def test_graph_records_search_queries_from_researcher() -> None:
    llm = _ScriptedLLM([_plan_json(["query a", "query b"]), _grade_json(True, 0.9), "answer"])
    retrieval_service = _FakeRetrievalService([_doc("a:chunk:0000")])

    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)
    result = await compiled.ainvoke(create_initial_state("question", max_retries=2))

    assert result["search_queries"] == ["query a", "query b"]


# --- rewrite_query node, in isolation --------------------------------------


@pytest.mark.unit
async def test_rewrite_query_node_updates_search_queries() -> None:
    from querymesh.agents import CitationAgent, GraderAgent, RetrieverAgent, SynthesizerAgent
    from querymesh.agents.researcher import ResearcherAgent
    from querymesh.graph.nodes import GraphNodes

    llm = _ScriptedLLM([_plan_json(["broadened query"])])
    nodes = GraphNodes(
        researcher=ResearcherAgent(llm),  # type: ignore[arg-type]
        retriever=RetrieverAgent(_FakeRetrievalService([])),  # type: ignore[arg-type]
        grader=GraderAgent(llm),  # type: ignore[arg-type]
        synthesizer=SynthesizerAgent(llm),  # type: ignore[arg-type]
        citation=CitationAgent(FakeLLM()),
    )

    state = create_initial_state("original question")
    state["search_queries"] = ["old narrow query"]

    update = await nodes.rewrite_query(state)

    assert update["search_queries"] == ["broadened query"]


# --- Retry loop -------------------------------------------------------------


@pytest.mark.unit
async def test_graph_retries_when_grading_yields_no_evidence_then_succeeds() -> None:
    """First pass: grader rejects everything -> evidence empty -> synthesizer
    flags insufficient -> supervisor routes to retry -> rewrite_query produces
    new search_queries -> second pass: grader accepts -> finishes.
    """
    llm = _ScriptedLLM(
        [
            _plan_json(["query one"]),  # research (1st pass)
            _grade_json(False, 0.1),  # grade (1st pass) -> rejected, evidence empty
            "no evidence available",  # synthesize (1st pass)
            _plan_json(["query one, broadened"]),  # rewrite_query (before retry pass)
            _grade_json(True, 0.9),  # grade (2nd pass, after retry) -> accepted
            "Final grounded answer after retry.",  # synthesize (2nd pass)
        ]
    )
    retrieval_service = _FakeRetrievalService([_doc("a:chunk:0000")])

    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)
    result = await compiled.ainvoke(create_initial_state("question", max_retries=2))

    assert result["retry_count"] == 1
    assert result["answer"] == "Final grounded answer after retry."
    assert len(result["citations"]) == 1
    assert retrieval_service.call_count == 2  # retrieval ran twice
    assert result["search_queries"] == ["query one, broadened"]  # rewritten, not resent


@pytest.mark.unit
async def test_graph_terminates_deterministically_when_evidence_never_sufficient() -> None:
    """Grader always rejects everything -> retries exhaust at max_retries,
    then the graph finishes anyway (never loops forever).
    """
    always_reject_llm = _ScriptedLLM([_plan_json(["q"]), _grade_json(False, 0.0), "no evidence"])
    # Empty docs -> grading never finds anything relevant, every pass.
    retrieval_service = _FakeRetrievalService([])

    compiled = compile_workflow(llm=always_reject_llm, retrieval_service=retrieval_service)
    result = await compiled.ainvoke(create_initial_state("question", max_retries=2))

    assert result["retry_count"] == 2  # capped at max_retries
    assert result["evidence"] == []
    # Graph terminated (didn't hang/raise a recursion error) -- reaching
    # this assertion at all is the real proof.
