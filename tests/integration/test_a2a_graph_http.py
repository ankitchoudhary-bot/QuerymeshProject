# 

"""End-to-end test: the full compiled LangGraph workflow, with every node
dispatching via `A2AClient` to **real, separately-running FastMCP HTTP
servers** — five independent `uvicorn`-backed processes (well, asyncio
tasks standing in for separate OS processes, exactly like
`test_http_transport.py`'s harness — real TCP sockets, real HTTP framing,
real MCP protocol, the only thing not literally different is which OS
process the bytes happen to originate from).

This is the test that was missing: `GraphNodes`/`compile_workflow` already
accept an `a2a_client` and dispatch through it (querymesh/graph/nodes.py),
and `MCPHttpTransport` was already tested in isolation
(test_http_transport.py) and `A2AClient.send()` was tested against
in-process handlers (test_a2a_integration.py) — but nothing proved the
*combination*: a real graph run, through `A2AClient`, over real HTTP, to
real remote agent processes, producing a real grounded answer.
"""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
from fastmcp import FastMCP

from querymesh.a2a import A2AClient
from querymesh.agents.citation import CitationAgent
from querymesh.agents.grader import GraderAgent
from querymesh.agents.researcher import ResearcherAgent
from querymesh.agents.synthesizer import SynthesizerAgent
from querymesh.graph import compile_workflow, create_initial_state
from querymesh.llm.fake import FakeLLM
from querymesh.mcp.servers import (
    build_citation_server,
    build_grader_server,
    build_research_server,
    build_synthesis_server,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RunningServer:
    """Starts a real FastMCP HTTP server as a background asyncio task and
    tears it down cleanly. Same harness as test_http_transport.py.
    """

    def __init__(self, mcp: FastMCP) -> None:
        self.mcp = mcp
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> "_RunningServer":
        self._task = asyncio.create_task(
            self.mcp.run_async(transport="http", host="127.0.0.1", port=self.port, show_banner=False)
        )
        await asyncio.sleep(1.0)  # real startup: let uvicorn's socket actually bind
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        assert self._task is not None
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass


def _grade_json(relevant: bool = True, score: float = 0.9) -> str:
    return json.dumps({"relevant": relevant, "score": score, "reason": "test"})


class _FakeRetrievalServiceForNonA2ARetriever:
    """`compile_workflow` always constructs a `RetrieverAgent` (it needs a
    `retrieval_service` regardless of whether A2A is used — see the
    docstring on `compile_workflow`), but in this test the *retrieve* node
    dispatches through the in-process handler path (registered below)
    rather than a real remote `retrieval-agent` MCP server, since Phase
    16/17's `retrieval_server.py` return shape is already covered by
    `test_http_transport.py`'s `test_transport_wraps_list_returning_tool_as_items`
    — this test's job is proving the *graph* dispatches correctly to
    *multiple simultaneous real remote agents*, not re-proving retrieval's
    list-wrapping again.
    """

    async def retrieve(self, query: str, top_k: int = 10, filters=None):
        raise AssertionError("should never be called: retrieve node uses the A2A handler below")


@pytest.mark.integration
async def test_graph_runs_end_to_end_dispatching_to_real_remote_mcp_servers() -> None:
    """Researcher, grader, synthesizer, and citation each run as their own
    real FastMCP HTTP server (separate asyncio tasks/ports, real TCP);
    retrieval uses an in-process A2A handler (proving in-process and
    remote recipients coexist on the same `A2AClient`, per its own
    docstring). The compiled graph dispatches every node through
    `A2AClient.send()` -> real HTTP -> real agent -> real response, and
    produces a genuine grounded answer with citations.
    """
    llm = FakeLLM(response_text=_grade_json())  # used by research/grade servers
    grader_llm = FakeLLM(response_text=_grade_json())
    synth_llm = FakeLLM(response_text="Attention mechanisms evolved toward efficiency.")

    researcher_server = build_research_server(ResearcherAgent(llm))
    grader_server = build_grader_server(GraderAgent(grader_llm))
    synthesis_server = build_synthesis_server(SynthesizerAgent(synth_llm))
    citation_server = build_citation_server(CitationAgent(FakeLLM()))

    async with (
        _RunningServer(researcher_server) as researcher_proc,
        _RunningServer(grader_server) as grader_proc,
        _RunningServer(synthesis_server) as synthesis_proc,
        _RunningServer(citation_server) as citation_proc,
    ):
        a2a_client = A2AClient()
        a2a_client.register_remote_agent("researcher", url=researcher_proc.url, tool_name="search_papers")
        a2a_client.register_remote_agent("grader", url=grader_proc.url, tool_name="grade_documents")
        a2a_client.register_remote_agent(
            "synthesizer", url=synthesis_proc.url, tool_name="generate_answer"
        )
        a2a_client.register_remote_agent("citation", url=citation_proc.url, tool_name="verify_citations")

        async def retrieve_handler(payload: dict) -> dict:
            return {
                "items": [
                    {
                        "document": {
                            "chunk_id": "2608.13558v1:chunk:0000",
                            "arxiv_id": "2608.13558v1",
                            "title": "Attention Evolutions",
                            "text": "Attention mechanisms allow models to weigh input relevance.",
                        },
                        "score": 5.0,
                        "rank": 1,
                        "retrieval_method": "hybrid_rrf",
                    }
                ]
            }

        a2a_client.register_handler("retriever", retrieve_handler)

        compiled = compile_workflow(
            llm=llm,
            retrieval_service=_FakeRetrievalServiceForNonA2ARetriever(),  # type: ignore[arg-type]
            a2a_client=a2a_client,
        )

        result = await compiled.ainvoke(
            create_initial_state("How have attention mechanisms evolved?", max_retries=1)
        )

    # The graph produced a real answer via four separate real HTTP agent
    # processes plus one in-process A2A handler — not a crash, not an
    # empty fallback.
    assert result["errors"] == []
    assert "efficiency" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["citations"][0].chunk_id == "2608.13558v1:chunk:0000"
    assert len(result["evidence"]) == 1


@pytest.mark.integration
async def test_graph_via_a2a_reports_error_when_a_remote_agent_is_unreachable() -> None:
    """A real network failure (nothing listening on the port) surfaces as
    a node-level error in `state["errors"]`, per the graph's existing
    per-node failure-isolation design (Phase 13) — it doesn't crash the
    whole run just because one remote agent process is down; `research()`
    falls back to using the raw query as its search query and the graph
    continues (this is `ResearcherAgent`'s own documented fallback
    contract, exercised here through a real connection failure rather
    than a mocked one).
    """
    dead_port = _free_port()  # nothing is listening here

    a2a_client = A2AClient(max_retries=0, timeout_seconds=2.0)
    a2a_client.register_remote_agent(
        "researcher", url=f"http://127.0.0.1:{dead_port}/mcp", tool_name="search_papers"
    )

    retrieve_calls: list[list[str]] = []

    async def retrieve_handler(payload: dict) -> dict:
        retrieve_calls.append(payload["queries"])
        return {"items": []}  # no evidence found; keeps the rest of the graph simple

    a2a_client.register_handler("retriever", retrieve_handler)

    llm = FakeLLM(response_text="unused")
    compiled = compile_workflow(
        llm=llm,
        retrieval_service=_FakeRetrievalServiceForNonA2ARetriever(),  # type: ignore[arg-type]
        a2a_client=a2a_client,
    )

    result = await compiled.ainvoke(create_initial_state("a query", max_retries=0))

    # The connection failure is captured, not swallowed silently or raised
    # out of the graph run.
    assert any("research" in e for e in result["errors"])
    # ResearcherAgent's graceful fallback: falls back to the raw query.
    assert result["search_queries"] == ["a query"]
    # The graph still proceeded to the next node with that fallback data,
    # rather than aborting the whole run over one remote agent's failure.
    assert retrieve_calls == [["a query"]]
