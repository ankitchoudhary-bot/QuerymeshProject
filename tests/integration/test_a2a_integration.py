# """Integration test: A2AClient dispatching to real agent methods (wrapped
# as handlers), not toy functions — exercises the real envelope-building +
# dispatch + agent-execution path together, plus the AgentRegistry from
# Phase 15 supplying the handler lookup.
# """

# from __future__ import annotations

# import json

# import pytest

# from querymesh.a2a import A2AClient, build_default_registry
# from querymesh.agents import CitationAgent, GraderAgent, ResearcherAgent, RetrieverAgent, SynthesizerAgent
# from querymesh.llm.fake import FakeLLM
# from querymesh.retrieval.models import RetrievedDocument


# class _FakeRetrievalService:
#     async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
#         return [
#             RetrievedDocument(
#                 document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "attention text"},
#                 score=5.0,
#                 rank=1,
#                 retrieval_method="hybrid_rrf",
#             )
#         ]


# @pytest.mark.integration
# async def test_a2a_client_dispatches_to_real_researcher_agent() -> None:
#     llm = FakeLLM(response_text=json.dumps({"search_queries": ["q1", "q2"], "reasoning": "r"}))
#     researcher = ResearcherAgent(llm)

#     async def researcher_handler(payload: dict) -> dict:
#         plan = await researcher.plan(payload["query"])
#         return plan.model_dump()

#     client = A2AClient()
#     client.register_handler("researcher", researcher_handler)

#     response = await client.send(
#         sender="supervisor", recipient="researcher", task="plan", payload={"query": "what is attention?"}
#     )

#     assert response.status == "response"
#     assert response.payload["search_queries"] == ["q1", "q2"]


# @pytest.mark.integration
# async def test_a2a_client_dispatches_through_registry_discovered_agents() -> None:
#     """Combines Phase 15 (AgentRegistry) with Phase 17 (A2AClient): the
#     registry supplies the real agent instances, A2AClient dispatches to
#     handlers wrapping them, discovered by capability rather than hardcoded.
#     """
#     llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}))
#     registry = build_default_registry(
#         ResearcherAgent(llm),
#         RetrieverAgent(_FakeRetrievalService()),  # type: ignore[arg-type]
#         GraderAgent(llm),
#         SynthesizerAgent(llm),
#         CitationAgent(),
#     )

#     a2a_client = A2AClient()

#     grader_agents = registry.discover("grading")
#     assert len(grader_agents) == 1
#     grader = grader_agents[0]

#     async def grader_handler(payload: dict) -> dict:
#         docs = [RetrievedDocument(**d) for d in payload["documents"]]
#         graded = await grader.grade_and_filter(payload["query"], docs)
#         return {"graded": [g.model_dump() for g in graded]}

#     a2a_client.register_handler("grader", grader_handler)

#     doc = RetrievedDocument(
#         document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "t"},
#         score=1.0,
#         rank=1,
#         retrieval_method="hybrid_rrf",
#     )
#     response = await a2a_client.send(
#         sender="supervisor",
#         recipient="grader",
#         task="grade",
#         payload={"query": "q", "documents": [doc.model_dump()]},
#     )

#     assert response.status == "response"
#     assert len(response.payload["graded"]) == 1


# @pytest.mark.integration
# async def test_a2a_client_error_response_when_real_agent_raises() -> None:
#     """A real agent failure (LLM raising) surfaces as a structured error
#     response through the full A2A dispatch path, not a crash.
#     """

#     class _BrokenLLM:
#         model = "broken"

#         async def ainvoke(self, prompt: str):
#             raise RuntimeError("provider outage")

#     researcher = ResearcherAgent(_BrokenLLM())  # type: ignore[arg-type]

#     async def researcher_handler(payload: dict) -> dict:
#         plan = await researcher.plan(payload["query"])  # will raise, since ainvoke raises
#         return plan.model_dump()

#     client = A2AClient(max_retries=0)
#     client.register_handler("researcher", researcher_handler)

#     response = await client.send(
#         sender="supervisor", recipient="researcher", task="plan", payload={"query": "q"}
#     )

#     assert response.status == "error"
#     assert "provider outage" in response.error


"""Integration test: A2AClient dispatching to real agent methods (wrapped
as handlers), not toy functions — exercises the real envelope-building +
dispatch + agent-execution path together, plus the AgentRegistry from
Phase 15 supplying the handler lookup.
"""

from __future__ import annotations

import json

import pytest

from querymesh.a2a import A2AClient, build_default_registry
from querymesh.agents import CitationAgent, GraderAgent, ResearcherAgent, RetrieverAgent, SynthesizerAgent
from querymesh.llm.fake import FakeLLM
from querymesh.retrieval.models import RetrievedDocument


class _FakeRetrievalService:
    async def retrieve(self, query: str, top_k: int = 10, filters=None) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "attention text"},
                score=5.0,
                rank=1,
                retrieval_method="hybrid_rrf",
            )
        ]


@pytest.mark.integration
async def test_a2a_client_dispatches_to_real_researcher_agent() -> None:
    llm = FakeLLM(response_text=json.dumps({"search_queries": ["q1", "q2"], "reasoning": "r"}))
    researcher = ResearcherAgent(llm)

    async def researcher_handler(payload: dict) -> dict:
        plan = await researcher.plan(payload["query"])
        return plan.model_dump()

    client = A2AClient()
    client.register_handler("researcher", researcher_handler)

    response = await client.send(
        sender="supervisor", recipient="researcher", task="plan", payload={"query": "what is attention?"}
    )

    assert response.status == "response"
    assert response.payload["search_queries"] == ["q1", "q2"]


@pytest.mark.integration
async def test_a2a_client_dispatches_through_registry_discovered_agents() -> None:
    """Combines Phase 15 (AgentRegistry) with Phase 17 (A2AClient): the
    registry supplies the real agent instances, A2AClient dispatches to
    handlers wrapping them, discovered by capability rather than hardcoded.
    """
    llm = FakeLLM(response_text=json.dumps({"relevant": True, "score": 0.9, "reason": "ok"}))
    registry = build_default_registry(
        ResearcherAgent(llm),
        RetrieverAgent(_FakeRetrievalService()),  # type: ignore[arg-type]
        GraderAgent(llm),
        SynthesizerAgent(llm),
        CitationAgent(llm),
    )

    a2a_client = A2AClient()

    grader_agents = registry.discover("grading")
    assert len(grader_agents) == 1
    grader = grader_agents[0]

    async def grader_handler(payload: dict) -> dict:
        docs = [RetrievedDocument(**d) for d in payload["documents"]]
        graded = await grader.grade_and_filter(payload["query"], docs)
        return {"graded": [g.model_dump() for g in graded]}

    a2a_client.register_handler("grader", grader_handler)

    doc = RetrievedDocument(
        document={"chunk_id": "a:chunk:0000", "arxiv_id": "p1", "title": "T", "text": "t"},
        score=1.0,
        rank=1,
        retrieval_method="hybrid_rrf",
    )
    response = await a2a_client.send(
        sender="supervisor",
        recipient="grader",
        task="grade",
        payload={"query": "q", "documents": [doc.model_dump()]},
    )

    assert response.status == "response"
    assert len(response.payload["graded"]) == 1


@pytest.mark.integration
async def test_a2a_client_error_response_when_real_agent_raises() -> None:
    """A real agent failure (LLM raising) surfaces as a structured error
    response through the full A2A dispatch path, not a crash.
    """

    class _BrokenLLM:
        model = "broken"

        async def ainvoke(self, prompt: str):
            raise RuntimeError("provider outage")

    researcher = ResearcherAgent(_BrokenLLM())  # type: ignore[arg-type]

    async def researcher_handler(payload: dict) -> dict:
        plan = await researcher.plan(payload["query"])  # will raise, since ainvoke raises
        return plan.model_dump()

    client = A2AClient(max_retries=0)
    client.register_handler("researcher", researcher_handler)

    response = await client.send(
        sender="supervisor", recipient="researcher", task="plan", payload={"query": "q"}
    )

    assert response.status == "error"
    assert "provider outage" in response.error
