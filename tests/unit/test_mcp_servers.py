# """Phase 16 tests: each FastMCP server, exercised via an in-memory
# fastmcp.Client (no network transport, no SSE server actually listening).
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
# from querymesh.llm.fake import FakeLLM
# from querymesh.mcp.servers import (
#     build_citation_server,
#     build_grader_server,
#     build_research_server,
#     build_retrieval_server,
#     build_synthesis_server,
# )
# from querymesh.retrieval.models import RetrievedDocument


# def _rd_dict(chunk_id: str, arxiv_id: str = "p1", title: str = "T", text: str = "text") -> dict:
#     return RetrievedDocument(
#         document={"chunk_id": chunk_id, "arxiv_id": arxiv_id, "title": title, "text": text},
#         score=1.0,
#         rank=1,
#         retrieval_method="hybrid_rrf",
#     ).model_dump()


# # --- research_server --------------------------------------------------


# @pytest.mark.unit
# async def test_research_server_exposes_search_papers_tool() -> None:
#     llm = FakeLLM(response_text=json.dumps({"search_queries": ["a", "b"], "reasoning": "why"}))
#     server = build_research_server(ResearcherAgent(llm))

#     async with Client(server) as client:
#         tools = await client.list_tools()
#         assert "search_papers" in {t.name for t in tools}

#         result = await client.call_tool("search_papers", {"query": "what is attention?"})

#     assert result.data["search_queries"] == ["a", "b"]
#     assert result.data["reasoning"] == "why"


# # --- retrieval_server -------------------------------------------------


# class _FakeRetrievalService:
#     async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
#         return [
#             RetrievedDocument(
#                 document={"chunk_id": f"{query}:chunk:0000", "text": "t"},
#                 score=1.0,
#                 rank=1,
#                 retrieval_method="hybrid_rrf",
#             )
#         ]


# @pytest.mark.unit
# async def test_retrieval_server_exposes_retrieve_documents_tool() -> None:
#     server = build_retrieval_server(RetrieverAgent(_FakeRetrievalService()))  # type: ignore[arg-type]

#     async with Client(server) as client:
#         tools = await client.list_tools()
#         assert "retrieve_documents" in {t.name for t in tools}

#         result = await client.call_tool("retrieve_documents", {"queries": ["q1"], "top_k": 5})

#     assert len(result.data) == 1
#     assert result.data[0]["document"]["chunk_id"] == "q1:chunk:0000"


# # --- grader_server -----------------------------------------------------


# @pytest.mark.unit
# async def test_grader_server_exposes_grade_documents_tool() -> None:
#     llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "good"}))
#     server = build_grader_server(GraderAgent(llm))

#     async with Client(server) as client:
#         tools = await client.list_tools()
#         assert "grade_documents" in {t.name for t in tools}

#         result = await client.call_tool(
#             "grade_documents", {"query": "q", "documents": [_rd_dict("a:chunk:0000")]}
#         )

#     assert len(result.data) == 1
#     assert result.data[0]["chunk_id"] == "a:chunk:0000"
#     assert result.data[0]["grade"]["relevant"] is True


# @pytest.mark.unit
# async def test_grader_server_filters_irrelevant_documents() -> None:
#     llm = FakeLLM(response_text=json.dumps({"relevant": False, "score": 0.1, "reason": "off topic"}))
#     server = build_grader_server(GraderAgent(llm))

#     async with Client(server) as client:
#         result = await client.call_tool(
#             "grade_documents", {"query": "q", "documents": [_rd_dict("a:chunk:0000")]}
#         )

#     assert result.data == []


# # --- synthesis_server --------------------------------------------------


# @pytest.mark.unit
# async def test_synthesis_server_exposes_generate_answer_tool() -> None:
#     llm = FakeLLM(response_text="Grounded answer text.")
#     server = build_synthesis_server(SynthesizerAgent(llm))

#     async with Client(server) as client:
#         tools = await client.list_tools()
#         assert "generate_answer" in {t.name for t in tools}

#         result = await client.call_tool(
#             "generate_answer", {"query": "q", "evidence": [_rd_dict("a:chunk:0000")]}
#         )

#     assert result.data["answer"] == "Grounded answer text."
#     assert result.data["evidence_sufficient"] is True


# # --- citation_server ------------------------------------------------------


# @pytest.mark.unit
# async def test_citation_server_exposes_verify_citations_tool() -> None:
#     server = build_citation_server(CitationAgent())

#     async with Client(server) as client:
#         tools = await client.list_tools()
#         assert "verify_citations" in {t.name for t in tools}

#         result = await client.call_tool(
#             "verify_citations", {"answer": "the answer", "evidence": [_rd_dict("a:chunk:0000", arxiv_id="p1")]}
#         )

#     assert len(result.data) == 1
#     assert result.data[0]["chunk_id"] == "a:chunk:0000"
#     assert result.data[0]["arxiv_id"] == "p1"




"""Phase 16 tests: each FastMCP server, exercised via an in-memory
fastmcp.Client (no network transport, no SSE server actually listening).
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
from querymesh.llm.fake import FakeLLM
from querymesh.mcp.servers import (
    build_citation_server,
    build_grader_server,
    build_research_server,
    build_retrieval_server,
    build_synthesis_server,
)
from querymesh.retrieval.models import RetrievedDocument


def _rd_dict(chunk_id: str, arxiv_id: str = "p1", title: str = "T", text: str = "text") -> dict:
    return RetrievedDocument(
        document={"chunk_id": chunk_id, "arxiv_id": arxiv_id, "title": title, "text": text},
        score=1.0,
        rank=1,
        retrieval_method="hybrid_rrf",
    ).model_dump()


# --- research_server --------------------------------------------------


@pytest.mark.unit
async def test_research_server_exposes_search_papers_tool() -> None:
    llm = FakeLLM(response_text=json.dumps({"search_queries": ["a", "b"], "reasoning": "why"}))
    server = build_research_server(ResearcherAgent(llm))

    async with Client(server) as client:
        tools = await client.list_tools()
        assert "search_papers" in {t.name for t in tools}

        result = await client.call_tool("search_papers", {"query": "what is attention?"})

    assert result.data["search_queries"] == ["a", "b"]
    assert result.data["reasoning"] == "why"


@pytest.mark.unit
async def test_research_server_exposes_rewrite_search_queries_tool() -> None:
    llm = FakeLLM(response_text=json.dumps({"search_queries": ["broader query"], "reasoning": "widened"}))
    server = build_research_server(ResearcherAgent(llm))

    async with Client(server) as client:
        tools = await client.list_tools()
        assert "rewrite_search_queries" in {t.name for t in tools}

        result = await client.call_tool(
            "rewrite_search_queries",
            {"query": "what is attention?", "previous_queries": ["narrow query"]},
        )

    assert result.data["search_queries"] == ["broader query"]
    assert result.data["reasoning"] == "widened"


# --- retrieval_server -------------------------------------------------


class _FakeRetrievalService:
    async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document={"chunk_id": f"{query}:chunk:0000", "text": "t"},
                score=1.0,
                rank=1,
                retrieval_method="hybrid_rrf",
            )
        ]


@pytest.mark.unit
async def test_retrieval_server_exposes_retrieve_documents_tool() -> None:
    server = build_retrieval_server(RetrieverAgent(_FakeRetrievalService()))  # type: ignore[arg-type]

    async with Client(server) as client:
        tools = await client.list_tools()
        assert "retrieve_documents" in {t.name for t in tools}

        result = await client.call_tool("retrieve_documents", {"queries": ["q1"], "top_k": 5})

    assert len(result.data) == 1
    assert result.data[0]["document"]["chunk_id"] == "q1:chunk:0000"


# --- grader_server -----------------------------------------------------


@pytest.mark.unit
async def test_grader_server_exposes_grade_documents_tool() -> None:
    llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "good"}))
    server = build_grader_server(GraderAgent(llm))

    async with Client(server) as client:
        tools = await client.list_tools()
        assert "grade_documents" in {t.name for t in tools}

        result = await client.call_tool(
            "grade_documents", {"query": "q", "documents": [_rd_dict("a:chunk:0000")]}
        )

    assert len(result.data) == 1
    assert result.data[0]["chunk_id"] == "a:chunk:0000"
    assert result.data[0]["grade"]["relevant"] is True


@pytest.mark.unit
async def test_grader_server_filters_irrelevant_documents() -> None:
    llm = FakeLLM(response_text=json.dumps({"relevant": False, "score": 0.1, "reason": "off topic"}))
    server = build_grader_server(GraderAgent(llm))

    async with Client(server) as client:
        result = await client.call_tool(
            "grade_documents", {"query": "q", "documents": [_rd_dict("a:chunk:0000")]}
        )

    assert result.data == []


# --- synthesis_server --------------------------------------------------


@pytest.mark.unit
async def test_synthesis_server_exposes_generate_answer_tool() -> None:
    llm = FakeLLM(response_text="Grounded answer text.")
    server = build_synthesis_server(SynthesizerAgent(llm))

    async with Client(server) as client:
        tools = await client.list_tools()
        assert "generate_answer" in {t.name for t in tools}

        result = await client.call_tool(
            "generate_answer", {"query": "q", "evidence": [_rd_dict("a:chunk:0000")]}
        )

    assert result.data["answer"] == "Grounded answer text."
    assert result.data["evidence_sufficient"] is True


# --- citation_server ------------------------------------------------------


@pytest.mark.unit
async def test_citation_server_exposes_verify_citations_tool() -> None:
    mapping_llm = FakeLLM(
        response_text=json.dumps({"mappings": [{"sentence_index": 0, "chunk_ids": ["a:chunk:0000"]}]})
    )
    server = build_citation_server(CitationAgent(mapping_llm))

    async with Client(server) as client:
        tools = await client.list_tools()
        assert "verify_citations" in {t.name for t in tools}

        result = await client.call_tool(
            "verify_citations", {"answer": "the answer", "evidence": [_rd_dict("a:chunk:0000", arxiv_id="p1")]}
        )

    assert len(result.data) == 1
    assert result.data[0]["chunk_id"] == "a:chunk:0000"
    assert result.data[0]["arxiv_id"] == "p1"
