# """Phase 11 tests: each of the five specialized agents, independently
# testable against FakeLLM / fake retrieval doubles.
# """

# from __future__ import annotations

# import json

# import pytest

# from querymesh.agents import CitationAgent, GraderAgent, ResearcherAgent, RetrieverAgent, SynthesizerAgent
# from querymesh.llm.fake import FakeLLM
# from querymesh.retrieval.models import RetrievedDocument


# def _rd(chunk_id: str, arxiv_id: str, title: str, text: str, score: float = 1.0) -> RetrievedDocument:
#     return RetrievedDocument(
#         document={"chunk_id": chunk_id, "arxiv_id": arxiv_id, "title": title, "text": text},
#         score=score,
#         rank=1,
#         retrieval_method="hybrid_rrf",
#     )


# # --- ResearcherAgent -----------------------------------------------------


# @pytest.mark.unit
# async def test_researcher_parses_valid_json_plan() -> None:
#     llm = FakeLLM(
#         response_text=json.dumps(
#             {"search_queries": ["transformer attention", "efficient attention"], "reasoning": "two angles"}
#         )
#     )
#     agent = ResearcherAgent(llm=llm)

#     plan = await agent.plan("How has attention evolved?")

#     assert plan.search_queries == ["transformer attention", "efficient attention"]
#     assert plan.reasoning == "two angles"


# @pytest.mark.unit
# async def test_researcher_falls_back_to_raw_query_on_bad_json() -> None:
#     llm = FakeLLM(response_text="not valid json at all")
#     agent = ResearcherAgent(llm=llm)

#     plan = await agent.plan("How has attention evolved?")

#     assert plan.search_queries == ["How has attention evolved?"]


# @pytest.mark.unit
# async def test_researcher_handles_code_fenced_json() -> None:
#     llm = FakeLLM(response_text='```json\n{"search_queries": ["a query"]}\n```')
#     agent = ResearcherAgent(llm=llm)

#     plan = await agent.plan("question")

#     assert plan.search_queries == ["a query"]


# @pytest.mark.unit
# async def test_researcher_rewrite_queries_produces_new_queries() -> None:
#     llm = FakeLLM(
#         response_text=json.dumps(
#             {"search_queries": ["broader query", "alternate terminology"], "reasoning": "widened scope"}
#         )
#     )
#     agent = ResearcherAgent(llm=llm)

#     plan = await agent.rewrite_queries("original question", previous_queries=["narrow query"])

#     assert plan.search_queries == ["broader query", "alternate terminology"]
#     assert plan.reasoning == "widened scope"


# @pytest.mark.unit
# async def test_researcher_rewrite_queries_falls_back_to_previous_on_bad_json() -> None:
#     llm = FakeLLM(response_text="not json")
#     agent = ResearcherAgent(llm=llm)

#     plan = await agent.rewrite_queries("original question", previous_queries=["prior a", "prior b"])

#     assert plan.search_queries == ["prior a", "prior b"]


# @pytest.mark.unit
# async def test_researcher_rewrite_queries_falls_back_to_user_query_when_no_previous() -> None:
#     llm = FakeLLM(response_text="not json")
#     agent = ResearcherAgent(llm=llm)

#     plan = await agent.rewrite_queries("original question", previous_queries=[])

#     assert plan.search_queries == ["original question"]


# # --- RetrieverAgent --------------------------------------------------------


# class _FakeRetrievalService:
#     def __init__(self, results_by_query: dict[str, list[RetrievedDocument]]) -> None:
#         self._results_by_query = results_by_query
#         self.calls: list[str] = []

#     async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
#         self.calls.append(query)
#         return self._results_by_query.get(query, [])


# @pytest.mark.unit
# async def test_retriever_agent_merges_multiple_queries() -> None:
#     service = _FakeRetrievalService(
#         {
#             "query one": [_rd("a", "p1", "T", "text a", score=5.0)],
#             "query two": [_rd("b", "p2", "T", "text b", score=3.0)],
#         }
#     )
#     agent = RetrieverAgent(retrieval_service=service)  # type: ignore[arg-type]

#     results = await agent.retrieve(["query one", "query two"])

#     assert {r.chunk_id for r in results} == {"a", "b"}
#     assert service.calls == ["query one", "query two"]


# @pytest.mark.unit
# async def test_retriever_agent_deduplicates_keeping_highest_score() -> None:
#     service = _FakeRetrievalService(
#         {
#             "query one": [_rd("a", "p1", "T", "text", score=2.0)],
#             "query two": [_rd("a", "p1", "T", "text", score=9.0)],
#         }
#     )
#     agent = RetrieverAgent(retrieval_service=service)  # type: ignore[arg-type]

#     results = await agent.retrieve(["query one", "query two"])

#     assert len(results) == 1
#     assert results[0].score == 9.0


# @pytest.mark.unit
# async def test_retriever_agent_sorts_by_score_descending() -> None:
#     service = _FakeRetrievalService(
#         {"q": [_rd("low", "p", "T", "t", score=1.0), _rd("high", "p", "T", "t", score=9.0)]}
#     )
#     agent = RetrieverAgent(retrieval_service=service)  # type: ignore[arg-type]

#     results = await agent.retrieve(["q"])

#     assert [r.chunk_id for r in results] == ["high", "low"]


# # --- GraderAgent -----------------------------------------------------------


# @pytest.mark.unit
# async def test_grader_agent_parses_grade() -> None:
#     llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "on topic"}))
#     agent = GraderAgent(llm=llm)
#     doc = _rd("a", "p1", "T", "relevant passage text")

#     graded = await agent.grade_one("query", doc)

#     assert graded.grade.relevant is True
#     assert graded.grade.score == 0.9
#     assert graded.chunk_id == "a"


# @pytest.mark.unit
# async def test_grader_agent_bad_json_grades_as_not_relevant() -> None:
#     llm = FakeLLM(response_text="garbage output")
#     agent = GraderAgent(llm=llm)
#     doc = _rd("a", "p1", "T", "text")

#     graded = await agent.grade_one("query", doc)

#     assert graded.grade.relevant is False
#     assert graded.grade.score == 0.0


# @pytest.mark.unit
# async def test_grade_and_filter_drops_irrelevant_and_low_score() -> None:
#     class _ScriptedLLM:
#         model = "scripted"

#         def __init__(self, responses: list[str]) -> None:
#             self._responses = iter(responses)

#         async def ainvoke(self, prompt: str):
#             from querymesh.llm.base import LLMResponse

#             return LLMResponse(text=next(self._responses), model=self.model)

#     llm = _ScriptedLLM(
#         [
#             json.dumps({"relevant": True, "score": 0.9, "reason": "great"}),
#             json.dumps({"relevant": False, "score": 0.1, "reason": "off topic"}),
#             json.dumps({"relevant": True, "score": 0.2, "reason": "weak match"}),  # below threshold
#         ]
#     )
#     agent = GraderAgent(llm=llm, relevance_threshold=0.5)  # type: ignore[arg-type]
#     docs = [_rd("a", "p", "T", "t"), _rd("b", "p", "T", "t"), _rd("c", "p", "T", "t")]

#     filtered = await agent.grade_and_filter("query", docs)

#     assert [g.chunk_id for g in filtered] == ["a"]


# # --- SynthesizerAgent ------------------------------------------------------


# @pytest.mark.unit
# async def test_synthesizer_produces_answer_with_evidence() -> None:
#     llm = FakeLLM(response_text="Attention evolved from full to sparse mechanisms.")
#     agent = SynthesizerAgent(llm=llm)
#     evidence = [_rd("a", "p1", "Attention Paper", "some evidence text")]

#     result = await agent.synthesize("How did attention evolve?", evidence)

#     assert "Attention evolved" in result.answer
#     assert result.evidence_sufficient is True


# @pytest.mark.unit
# async def test_synthesizer_flags_insufficient_evidence_when_none_given() -> None:
#     llm = FakeLLM(response_text="I don't have enough evidence to answer.")
#     agent = SynthesizerAgent(llm=llm)

#     result = await agent.synthesize("Unanswerable question", evidence=[])

#     assert result.evidence_sufficient is False


# # --- CitationAgent -----------------------------------------------------------


# @pytest.mark.unit
# def test_citation_agent_builds_one_citation_per_evidence_chunk() -> None:
#     agent = CitationAgent()
#     evidence = [
#         _rd("a:chunk:0000", "2608.13558v1", "Attention Evolutions", "text a"),
#         _rd("b:chunk:0001", "2608.99999v1", "Other Paper", "text b"),
#     ]

#     citations = agent.build_citations("The answer.", evidence)

#     assert len(citations) == 2
#     assert citations[0].arxiv_id == "2608.13558v1"
#     assert citations[0].chunk_id == "a:chunk:0000"


# @pytest.mark.unit
# def test_citation_agent_verify_drops_unbacked_citations() -> None:
#     agent = CitationAgent()
#     evidence = [_rd("real:chunk:0000", "p1", "T", "text")]

#     from querymesh.agents.models import Citation

#     citations = [
#         Citation(arxiv_id="p1", title="T", chunk_id="real:chunk:0000", claim="ok"),
#         Citation(arxiv_id="fake", title="Invented", chunk_id="invented:chunk:9999", claim="bad"),
#     ]

#     verified = agent.verify_citations(citations, evidence)

#     assert len(verified) == 1
#     assert verified[0].chunk_id == "real:chunk:0000"


# @pytest.mark.unit
# def test_citation_agent_no_evidence_yields_no_citations() -> None:
#     agent = CitationAgent()
#     citations = agent.build_citations("answer", evidence=[])
#     assert citations == []





"""Phase 11 tests: each of the five specialized agents, independently
testable against FakeLLM / fake retrieval doubles.
"""

from __future__ import annotations

import json

import pytest

from querymesh.agents import CitationAgent, GraderAgent, ResearcherAgent, RetrieverAgent, SynthesizerAgent
from querymesh.llm.fake import FakeLLM
from querymesh.retrieval.models import RetrievedDocument


def _rd(chunk_id: str, arxiv_id: str, title: str, text: str, score: float = 1.0) -> RetrievedDocument:
    return RetrievedDocument(
        document={"chunk_id": chunk_id, "arxiv_id": arxiv_id, "title": title, "text": text},
        score=score,
        rank=1,
        retrieval_method="hybrid_rrf",
    )


# --- ResearcherAgent -----------------------------------------------------


@pytest.mark.unit
async def test_researcher_parses_valid_json_plan() -> None:
    llm = FakeLLM(
        response_text=json.dumps(
            {"search_queries": ["transformer attention", "efficient attention"], "reasoning": "two angles"}
        )
    )
    agent = ResearcherAgent(llm=llm)

    plan = await agent.plan("How has attention evolved?")

    assert plan.search_queries == ["transformer attention", "efficient attention"]
    assert plan.reasoning == "two angles"


@pytest.mark.unit
async def test_researcher_falls_back_to_raw_query_on_bad_json() -> None:
    llm = FakeLLM(response_text="not valid json at all")
    agent = ResearcherAgent(llm=llm)

    plan = await agent.plan("How has attention evolved?")

    assert plan.search_queries == ["How has attention evolved?"]


@pytest.mark.unit
async def test_researcher_handles_code_fenced_json() -> None:
    llm = FakeLLM(response_text='```json\n{"search_queries": ["a query"]}\n```')
    agent = ResearcherAgent(llm=llm)

    plan = await agent.plan("question")

    assert plan.search_queries == ["a query"]


@pytest.mark.unit
async def test_researcher_rewrite_queries_produces_new_queries() -> None:
    llm = FakeLLM(
        response_text=json.dumps(
            {"search_queries": ["broader query", "alternate terminology"], "reasoning": "widened scope"}
        )
    )
    agent = ResearcherAgent(llm=llm)

    plan = await agent.rewrite_queries("original question", previous_queries=["narrow query"])

    assert plan.search_queries == ["broader query", "alternate terminology"]
    assert plan.reasoning == "widened scope"


@pytest.mark.unit
async def test_researcher_rewrite_queries_falls_back_to_previous_on_bad_json() -> None:
    llm = FakeLLM(response_text="not json")
    agent = ResearcherAgent(llm=llm)

    plan = await agent.rewrite_queries("original question", previous_queries=["prior a", "prior b"])

    assert plan.search_queries == ["prior a", "prior b"]


@pytest.mark.unit
async def test_researcher_rewrite_queries_falls_back_to_user_query_when_no_previous() -> None:
    llm = FakeLLM(response_text="not json")
    agent = ResearcherAgent(llm=llm)

    plan = await agent.rewrite_queries("original question", previous_queries=[])

    assert plan.search_queries == ["original question"]


# --- RetrieverAgent --------------------------------------------------------


class _FakeRetrievalService:
    def __init__(self, results_by_query: dict[str, list[RetrievedDocument]]) -> None:
        self._results_by_query = results_by_query
        self.calls: list[str] = []

    async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        self.calls.append(query)
        return self._results_by_query.get(query, [])


@pytest.mark.unit
async def test_retriever_agent_merges_multiple_queries() -> None:
    service = _FakeRetrievalService(
        {
            "query one": [_rd("a", "p1", "T", "text a", score=5.0)],
            "query two": [_rd("b", "p2", "T", "text b", score=3.0)],
        }
    )
    agent = RetrieverAgent(retrieval_service=service)  # type: ignore[arg-type]

    results = await agent.retrieve(["query one", "query two"])

    assert {r.chunk_id for r in results} == {"a", "b"}
    assert service.calls == ["query one", "query two"]


@pytest.mark.unit
async def test_retriever_agent_deduplicates_keeping_highest_score() -> None:
    service = _FakeRetrievalService(
        {
            "query one": [_rd("a", "p1", "T", "text", score=2.0)],
            "query two": [_rd("a", "p1", "T", "text", score=9.0)],
        }
    )
    agent = RetrieverAgent(retrieval_service=service)  # type: ignore[arg-type]

    results = await agent.retrieve(["query one", "query two"])

    assert len(results) == 1
    assert results[0].score == 9.0


@pytest.mark.unit
async def test_retriever_agent_sorts_by_score_descending() -> None:
    service = _FakeRetrievalService(
        {"q": [_rd("low", "p", "T", "t", score=1.0), _rd("high", "p", "T", "t", score=9.0)]}
    )
    agent = RetrieverAgent(retrieval_service=service)  # type: ignore[arg-type]

    results = await agent.retrieve(["q"])

    assert [r.chunk_id for r in results] == ["high", "low"]


# --- GraderAgent -----------------------------------------------------------


@pytest.mark.unit
async def test_grader_agent_parses_grade() -> None:
    llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "on topic"}))
    agent = GraderAgent(llm=llm)
    doc = _rd("a", "p1", "T", "relevant passage text")

    graded = await agent.grade_one("query", doc)

    assert graded.grade.relevant is True
    assert graded.grade.score == 0.9
    assert graded.chunk_id == "a"


@pytest.mark.unit
async def test_grader_agent_bad_json_grades_as_not_relevant() -> None:
    llm = FakeLLM(response_text="garbage output")
    agent = GraderAgent(llm=llm)
    doc = _rd("a", "p1", "T", "text")

    graded = await agent.grade_one("query", doc)

    assert graded.grade.relevant is False
    assert graded.grade.score == 0.0


@pytest.mark.unit
async def test_grade_and_filter_drops_irrelevant_and_low_score() -> None:
    class _ScriptedLLM:
        model = "scripted"

        def __init__(self, responses: list[str]) -> None:
            self._responses = iter(responses)

        async def ainvoke(self, prompt: str):
            from querymesh.llm.base import LLMResponse

            return LLMResponse(text=next(self._responses), model=self.model)

    llm = _ScriptedLLM(
        [
            json.dumps({"relevant": True, "score": 0.9, "reason": "great"}),
            json.dumps({"relevant": False, "score": 0.1, "reason": "off topic"}),
            json.dumps({"relevant": True, "score": 0.2, "reason": "weak match"}),  # below threshold
        ]
    )
    agent = GraderAgent(llm=llm, relevance_threshold=0.5)  # type: ignore[arg-type]
    docs = [_rd("a", "p", "T", "t"), _rd("b", "p", "T", "t"), _rd("c", "p", "T", "t")]

    filtered = await agent.grade_and_filter("query", docs)

    assert [g.chunk_id for g in filtered] == ["a"]


# --- SynthesizerAgent ------------------------------------------------------


@pytest.mark.unit
async def test_synthesizer_produces_answer_with_evidence() -> None:
    llm = FakeLLM(response_text="Attention evolved from full to sparse mechanisms.")
    agent = SynthesizerAgent(llm=llm)
    evidence = [_rd("a", "p1", "Attention Paper", "some evidence text")]

    result = await agent.synthesize("How did attention evolve?", evidence)

    assert "Attention evolved" in result.answer
    assert result.evidence_sufficient is True


@pytest.mark.unit
async def test_synthesizer_flags_insufficient_evidence_when_none_given() -> None:
    llm = FakeLLM(response_text="I don't have enough evidence to answer.")
    agent = SynthesizerAgent(llm=llm)

    result = await agent.synthesize("Unanswerable question", evidence=[])

    assert result.evidence_sufficient is False


# --- CitationAgent -----------------------------------------------------------


def _mapping_response(*mappings: tuple[int, list[str]]) -> str:
    return json.dumps(
        {"mappings": [{"sentence_index": i, "chunk_ids": ids} for i, ids in mappings]}
    )


@pytest.mark.unit
async def test_citation_agent_maps_each_sentence_to_its_supporting_chunk() -> None:
    llm = FakeLLM(
        response_text=_mapping_response((0, ["a:chunk:0000"]), (1, ["b:chunk:0001"]))
    )
    agent = CitationAgent(llm)
    evidence = [
        _rd("a:chunk:0000", "2608.13558v1", "Attention Evolutions", "text a"),
        _rd("b:chunk:0001", "2608.99999v1", "Other Paper", "text b"),
    ]

    citations = await agent.build_citations(
        "Attention scales quadratically. Sparse variants reduce this cost.", evidence
    )

    assert len(citations) == 2
    assert citations[0].chunk_id == "a:chunk:0000"
    assert citations[0].claim == "Attention scales quadratically."
    assert citations[1].chunk_id == "b:chunk:0001"
    assert citations[1].claim == "Sparse variants reduce this cost."


@pytest.mark.unit
async def test_citation_agent_sentence_with_no_claim_gets_no_citation() -> None:
    llm = FakeLLM(response_text=_mapping_response((0, ["a:chunk:0000"]), (1, [])))
    agent = CitationAgent(llm)
    evidence = [_rd("a:chunk:0000", "p1", "T", "text a")]

    citations = await agent.build_citations("A real claim. In summary,", evidence)

    assert len(citations) == 1
    assert citations[0].claim == "A real claim."


@pytest.mark.unit
async def test_citation_agent_one_sentence_can_map_to_multiple_chunks() -> None:
    llm = FakeLLM(response_text=_mapping_response((0, ["a:chunk:0000", "b:chunk:0001"])))
    agent = CitationAgent(llm)
    evidence = [
        _rd("a:chunk:0000", "p1", "T1", "text a"),
        _rd("b:chunk:0001", "p2", "T2", "text b"),
    ]

    citations = await agent.build_citations("A well-supported claim.", evidence)

    assert {c.chunk_id for c in citations} == {"a:chunk:0000", "b:chunk:0001"}
    assert all(c.claim == "A well-supported claim." for c in citations)


@pytest.mark.unit
async def test_citation_agent_never_cites_a_hallucinated_chunk_id() -> None:
    llm = FakeLLM(response_text=_mapping_response((0, ["invented:chunk:9999"])))
    agent = CitationAgent(llm)
    evidence = [_rd("a:chunk:0000", "p1", "T", "text a")]

    citations = await agent.build_citations("A claim.", evidence)

    assert citations == []


@pytest.mark.unit
async def test_citation_agent_falls_back_to_coarse_citations_on_bad_json() -> None:
    llm = FakeLLM(response_text="not valid json")
    agent = CitationAgent(llm)
    evidence = [
        _rd("a:chunk:0000", "2608.13558v1", "Attention Evolutions", "text a"),
        _rd("b:chunk:0001", "2608.99999v1", "Other Paper", "text b"),
    ]

    citations = await agent.build_citations("The answer.", evidence)

    # Fallback: one citation per evidence chunk, claim = the whole answer.
    assert len(citations) == 2
    assert {c.chunk_id for c in citations} == {"a:chunk:0000", "b:chunk:0001"}
    assert all(c.claim == "The answer." for c in citations)


@pytest.mark.unit
async def test_citation_agent_passes_query_context_when_given() -> None:
    captured_prompts: list[str] = []

    class _CapturingLLM:
        model = "capturing"

        async def ainvoke(self, prompt: str):
            from querymesh.llm.base import LLMResponse

            captured_prompts.append(prompt)
            return LLMResponse(text=_mapping_response((0, ["a:chunk:0000"])), model=self.model)

    agent = CitationAgent(_CapturingLLM())  # type: ignore[arg-type]
    evidence = [_rd("a:chunk:0000", "p1", "T", "text a")]

    await agent.build_citations("A claim.", evidence, query="What is attention?")

    assert "What is attention?" in captured_prompts[0]


@pytest.mark.unit
async def test_citation_agent_rejects_none_llm() -> None:
    from querymesh.errors import AgentError

    with pytest.raises(AgentError):
        CitationAgent(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_citation_agent_verify_drops_unbacked_citations() -> None:
    agent = CitationAgent(FakeLLM())
    evidence = [_rd("real:chunk:0000", "p1", "T", "text")]

    from querymesh.agents.models import Citation

    citations = [
        Citation(arxiv_id="p1", title="T", chunk_id="real:chunk:0000", claim="ok"),
        Citation(arxiv_id="fake", title="Invented", chunk_id="invented:chunk:9999", claim="bad"),
    ]

    verified = agent.verify_citations(citations, evidence)

    assert len(verified) == 1
    assert verified[0].chunk_id == "real:chunk:0000"


@pytest.mark.unit
async def test_citation_agent_no_evidence_yields_no_citations() -> None:
    agent = CitationAgent(FakeLLM())
    citations = await agent.build_citations("answer", evidence=[])
    assert citations == []


@pytest.mark.unit
async def test_citation_agent_empty_answer_yields_no_citations() -> None:
    agent = CitationAgent(FakeLLM())
    evidence = [_rd("a:chunk:0000", "p1", "T", "text a")]
    citations = await agent.build_citations("   ", evidence)
    assert citations == []
