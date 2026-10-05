"""Builds and compiles the QueryMesh research graph.

 ┌───────────────┐
 │ SUPERVISOR    │  (supervisor_start)
 └───────┬───────┘
         ↓
     Researcher
         ↓
     Retriever
         ↓
      Grader
         ↓
    Synthesizer
         ↓
      Citation
         ↓
 ┌───────────────┐
 │ SUPERVISOR    │  (supervisor_review)
 └───────┬───────┘
      ┌──┴──┐
   retry   finish
      │       │
      ↓       ↓
 bump_retry   END
      │
      ↓
 rewrite_query
      │
      ↓
  (back to Retriever)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from querymesh.agents.citation import CitationAgent
from querymesh.agents.grader import GraderAgent
from querymesh.agents.researcher import ResearcherAgent
from querymesh.agents.retriever import RetrieverAgent
from querymesh.agents.synthesizer import SynthesizerAgent
from querymesh.graph.edges import FINISH, RETRY, route_after_supervisor
from querymesh.graph.nodes import GraphNodes
from querymesh.graph.state import ResearchState
from querymesh.graph.stream_events import StreamEvent, WORKFLOW_COMPLETED, WORKFLOW_STARTED, make_event
from querymesh.llm.base import LLM
from querymesh.observability.tracing import NoOpTracer, Tracer
from querymesh.retrieval.service import RetrievalService


def build_workflow(nodes: GraphNodes) -> StateGraph:
    """Construct the (uncompiled) graph from a `GraphNodes` instance."""
    graph: StateGraph = StateGraph(ResearchState)

    graph.add_node("supervisor_start", nodes.supervisor_start)
    graph.add_node("research", nodes.research)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("cite", nodes.cite)
    graph.add_node("supervisor_review", nodes.supervisor_review)
    graph.add_node("bump_retry", nodes.bump_retry)
    graph.add_node("rewrite_query", nodes.rewrite_query)

    graph.add_edge(START, "supervisor_start")
    graph.add_edge("supervisor_start", "research")
    graph.add_edge("research", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("grade", "synthesize")
    graph.add_edge("synthesize", "cite")
    graph.add_edge("cite", "supervisor_review")
    graph.add_conditional_edges(
        "supervisor_review",
        route_after_supervisor,
        {RETRY: "bump_retry", FINISH: END},
    )
    graph.add_edge("bump_retry", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")

    return graph


def compile_workflow(
    llm: LLM,
    retrieval_service: RetrievalService,
    grading_llm: LLM | None = None,
    tracer: Tracer | None = None,
) -> CompiledStateGraph:
    """Build agent instances and compile the full research graph."""
    nodes = GraphNodes(
        researcher=ResearcherAgent(llm),
        retriever=RetrieverAgent(retrieval_service),
        grader=GraderAgent(grading_llm or llm),
        synthesizer=SynthesizerAgent(llm),
        citation=CitationAgent(),
        tracer=tracer or NoOpTracer(),
    )
    return build_workflow(nodes).compile()


_DEFAULT_RECURSION_LIMIT = 50


async def run_research_workflow(
    compiled: CompiledStateGraph,
    initial_state: ResearchState,
    tracer: Tracer | None = None,
    recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
) -> dict[str, Any]:
    """Run a compiled graph inside a top-level "research_request" span.

    This is the root of the tracing hierarchy: "research_request" contains
    "researcher", "retrieval" (with "bm25"/"vector"/"rrf" children),
    "grader", "synthesis", and "citation" as nested spans, opened by the
    graph nodes and retrieval layer themselves.

    `recursion_limit` bounds LangGraph's own step counter as defense in
    depth beyond the retry-count logic in `edges.py` — belt and suspenders
    against a future routing bug creating an unintended cycle, per the
    "maximum graph recursion" production-safety requirement.
    """
    tracer = tracer or NoOpTracer()
    with tracer.span("research_request", query=initial_state["query"]):
        return await compiled.ainvoke(initial_state, config={"recursion_limit": recursion_limit})


async def run_research_workflow_streaming(
    compiled: CompiledStateGraph,
    initial_state: ResearchState,
    tracer: Tracer | None = None,
    recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
) -> AsyncIterator[StreamEvent]:
    """Run a compiled graph, yielding structured `StreamEvent`s as the
    workflow progresses: `workflow_started`, per-node `agent_started`/
    `retrieval_started`/`retrieval_completed`/`grading_completed`/
    `generation_started`/`token`/`citation` events (emitted by the nodes
    themselves via LangGraph's custom stream mode), and finally
    `workflow_completed` (or `error` if the run raises).
    """
    tracer = tracer or NoOpTracer()
    yield make_event(WORKFLOW_STARTED, "Starting research request", query=initial_state["query"])

    try:
        with tracer.span("research_request", query=initial_state["query"]):
            final_state: dict[str, Any] = dict(initial_state)
            async for stream_mode, event in compiled.astream(
                initial_state,
                stream_mode=["updates", "custom"],
                config={"recursion_limit": recursion_limit},
            ):
                if stream_mode == "custom":
                    yield event
                elif stream_mode == "updates":
                    for _node_name, update in event.items():
                        if update:
                            final_state.update(update)
    except Exception as exc:  # noqa: BLE001 - surface as a stream event, don't crash the consumer
        yield make_event("error", str(exc))
        return

    yield make_event(
        WORKFLOW_COMPLETED,
        "Research request complete",
        answer=final_state.get("answer", ""),
        citation_count=len(final_state.get("citations", [])),
        retry_count=final_state.get("retry_count", 0),
    )
