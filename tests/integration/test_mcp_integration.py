# """Integration test: all five FastMCP servers, called in sequence through
# real `fastmcp.Client` tool calls, each server's output feeding the next
# server's input — exercising the real MCP protocol round-trip across the
# whole agent chain, not just one tool at a time.
# """

# from __future__ import annotations

# import json

# import pytest
# from fastmcp import Client

# from querymesh.agents.citation import CitationAgent
# from querymesh.agents.grader import GraderAgent
# from querymesh.agents.researcher import ResearcherAgent
# from querymesh.agents.retriever import RetrieverAgent
# from querymesh.agents.synthesizer import SynthesizerAgent
# from querymesh.llm.base import LLMResponse
# from querymesh.mcp.servers import (
#     build_citation_server,
#     build_grader_server,
#     build_research_server,
#     build_retrieval_server,
#     build_synthesis_server,
# )
# from querymesh.retrieval.models import RetrievedDocument


# class _ScriptedLLM:
#     model = "scripted"

#     def __init__(self, responses: list[str]) -> None:
#         self._responses = responses
#         self._index = 0

#     async def ainvoke(self, prompt: str) -> LLMResponse:
#         text = self._responses[min(self._index, len(self._responses) - 1)]
#         self._index += 1
#         return LLMResponse(text=text, model=self.model)


# class _FakeRetrievalService:
#     async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
#         return [
#             RetrievedDocument(
#                 document={
#                     "chunk_id": "2608.13558v1:chunk:0000",
#                     "arxiv_id": "2608.13558v1",
#                     "title": "Attention Evolutions",
#                     "text": "Attention mechanisms have evolved significantly.",
#                 },
#                 score=5.0,
#                 rank=1,
#                 retrieval_method="hybrid_rrf",
#             )
#         ]


# @pytest.mark.integration
# async def test_full_mcp_agent_chain_end_to_end() -> None:
#     """search_papers -> retrieve_documents -> grade_documents ->
#     generate_answer -> verify_citations, each a real MCP tool call, output
#     of each step feeding the next.
#     """
#     llm = _ScriptedLLM(
#         [
#             json.dumps({"search_queries": ["attention mechanisms"], "reasoning": "plan"}),
#             json.dumps({"relevant": True, "score": 0.9, "reason": "on topic"}),
#             "Attention mechanisms have evolved from full to sparse forms.",
#         ]
#     )

#     research_server = build_research_server(ResearcherAgent(llm))  # type: ignore[arg-type]
#     retrieval_server = build_retrieval_server(RetrieverAgent(_FakeRetrievalService()))  # type: ignore[arg-type]
#     grader_server = build_grader_server(GraderAgent(llm))  # type: ignore[arg-type]
#     synthesis_server = build_synthesis_server(SynthesizerAgent(llm))  # type: ignore[arg-type]
#     citation_server = build_citation_server(CitationAgent())

#     # Step 1: research
#     async with Client(research_server) as client:
#         plan_result = await client.call_tool("search_papers", {"query": "How did attention evolve?"})
#     queries = plan_result.data["search_queries"]
#     assert queries == ["attention mechanisms"]

#     # Step 2: retrieval
#     async with Client(retrieval_server) as client:
#         retrieval_result = await client.call_tool("retrieve_documents", {"queries": queries, "top_k": 10})
#     documents = retrieval_result.data
#     assert len(documents) == 1

#     # Step 3: grading
#     async with Client(grader_server) as client:
#         grading_result = await client.call_tool(
#             "grade_documents", {"query": "How did attention evolve?", "documents": documents}
#         )
#     graded = grading_result.data
#     assert len(graded) == 1
#     assert graded[0]["grade"]["relevant"] is True

#     # Step 4: synthesis (evidence = the same documents, now grade-approved)
#     async with Client(synthesis_server) as client:
#         synthesis_result = await client.call_tool(
#             "generate_answer", {"query": "How did attention evolve?", "evidence": documents}
#         )
#     answer = synthesis_result.data["answer"]
#     assert "evolved" in answer

#     # Step 5: citation verification
#     async with Client(citation_server) as client:
#         citation_result = await client.call_tool(
#             "verify_citations", {"answer": answer, "evidence": documents}
#         )
#     citations = citation_result.data
#     assert len(citations) == 1
#     assert citations[0]["chunk_id"] == "2608.13558v1:chunk:0000"
#     assert citations[0]["arxiv_id"] == "2608.13558v1"



"""Integration test: all five FastMCP servers, called in sequence through
real `fastmcp.Client` tool calls, each server's output feeding the next
server's input — exercising the real MCP protocol round-trip across the
whole agent chain, not just one tool at a time.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

from querymesh.agents.citation import CitationAgent
from querymesh.agents.grader import GraderAgent
from querymesh.agents.researcher import ResearcherAgent
from querymesh.agents.retriever import RetrieverAgent
from querymesh.agents.synthesizer import SynthesizerAgent
from querymesh.llm.base import LLMResponse
from querymesh.llm.fake import FakeLLM
from querymesh.mcp.servers import (
    build_citation_server,
    build_grader_server,
    build_research_server,
    build_retrieval_server,
    build_synthesis_server,
)
from querymesh.retrieval.models import RetrievedDocument


class _ScriptedLLM:
    model = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    async def ainvoke(self, prompt: str) -> LLMResponse:
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return LLMResponse(text=text, model=self.model)


class _FakeRetrievalService:
    async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document={
                    "chunk_id": "2608.13558v1:chunk:0000",
                    "arxiv_id": "2608.13558v1",
                    "title": "Attention Evolutions",
                    "text": "Attention mechanisms have evolved significantly.",
                },
                score=5.0,
                rank=1,
                retrieval_method="hybrid_rrf",
            )
        ]


@pytest.mark.integration
async def test_full_mcp_agent_chain_end_to_end() -> None:
    """search_papers -> retrieve_documents -> grade_documents ->
    generate_answer -> verify_citations, each a real MCP tool call, output
    of each step feeding the next.
    """
    llm = _ScriptedLLM(
        [
            json.dumps({"search_queries": ["attention mechanisms"], "reasoning": "plan"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "on topic"}),
            "Attention mechanisms have evolved from full to sparse forms.",
        ]
    )

    research_server = build_research_server(ResearcherAgent(llm))  # type: ignore[arg-type]
    retrieval_server = build_retrieval_server(RetrieverAgent(_FakeRetrievalService()))  # type: ignore[arg-type]
    grader_server = build_grader_server(GraderAgent(llm))  # type: ignore[arg-type]
    synthesis_server = build_synthesis_server(SynthesizerAgent(llm))  # type: ignore[arg-type]
    citation_server = build_citation_server(CitationAgent(FakeLLM()))

    # Step 1: research
    async with Client(research_server) as client:
        plan_result = await client.call_tool("search_papers", {"query": "How did attention evolve?"})
    queries = plan_result.data["search_queries"]
    assert queries == ["attention mechanisms"]

    # Step 2: retrieval
    async with Client(retrieval_server) as client:
        retrieval_result = await client.call_tool("retrieve_documents", {"queries": queries, "top_k": 10})
    documents = retrieval_result.data
    assert len(documents) == 1

    # Step 3: grading
    async with Client(grader_server) as client:
        grading_result = await client.call_tool(
            "grade_documents", {"query": "How did attention evolve?", "documents": documents}
        )
    graded = grading_result.data
    assert len(graded) == 1
    assert graded[0]["grade"]["relevant"] is True

    # Step 4: synthesis (evidence = the same documents, now grade-approved)
    async with Client(synthesis_server) as client:
        synthesis_result = await client.call_tool(
            "generate_answer", {"query": "How did attention evolve?", "evidence": documents}
        )
    answer = synthesis_result.data["answer"]
    assert "evolved" in answer

    # Step 5: citation verification
    async with Client(citation_server) as client:
        citation_result = await client.call_tool(
            "verify_citations", {"answer": answer, "evidence": documents}
        )
    citations = citation_result.data
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "2608.13558v1:chunk:0000"
    assert citations[0]["arxiv_id"] == "2608.13558v1"
