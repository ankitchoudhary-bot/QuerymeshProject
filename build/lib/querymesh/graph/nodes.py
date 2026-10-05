"""LangGraph node functions.

Each node wraps one agent (or a small supervisor step) and does the
translation between `ResearchState` and that agent's own inputs/outputs.
Nodes are plain async methods on `GraphNodes` — a dataclass holding the
agent instances — so they stay independently callable/testable without a
compiled graph, exactly like the agents themselves.

Every node catches its own exceptions and records them in
`state["errors"]` rather than letting one node's failure crash the whole
graph run — callers can inspect `errors` afterward.

Each node opens a span named after its role in the observability
hierarchy (researcher/grader/synthesis/citation); "retrieval" (and its
bm25/vector/rrf children) is opened one layer down, inside
`RetrieverAgent` -> `RetrievalService`, so it nests correctly under
whatever span the caller of `compiled.ainvoke()` has open (see
`workflow.run_research_workflow`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph.config import get_stream_writer

from querymesh.agents.citation import CitationAgent
from querymesh.agents.grader import GraderAgent
from querymesh.agents.researcher import ResearcherAgent
from querymesh.agents.retriever import RetrieverAgent
from querymesh.agents.synthesizer import SynthesizerAgent
from querymesh.graph.state import ResearchState
from querymesh.graph.stream_events import (
    AGENT_STARTED,
    CITATION,
    GENERATION_STARTED,
    GRADING_COMPLETED,
    RETRIEVAL_COMPLETED,
    RETRIEVAL_STARTED,
    TOKEN,
    make_event,
)
from querymesh.observability.logging import get_logger
from querymesh.observability.tracing import NoOpTracer, Tracer
from querymesh.retrieval.models import RetrievedDocument

logger = get_logger(__name__)


def _safe_stream_writer():
    """`get_stream_writer()` raises when called fully outside a graph run
    (e.g. calling a node method directly in a unit test) rather than just
    returning a no-op — only the "no custom stream mode active, but still
    inside a run" case no-ops safely. Fall back to a no-op callable so
    nodes stay directly callable/testable, exactly as their docstring
    promises.
    """
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _event: None


@dataclass
class GraphNodes:
    researcher: ResearcherAgent
    retriever: RetrieverAgent
    grader: GraderAgent
    synthesizer: SynthesizerAgent
    citation: CitationAgent
    tracer: Tracer = field(default_factory=NoOpTracer)

    async def supervisor_start(self, state: ResearchState) -> dict:
        """Entry point. No-op today; a natural place for future routing
        logic (e.g. skipping straight to cache lookup) without touching
        every downstream node.
        """
        return {}

    async def research(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        writer(make_event(AGENT_STARTED, "Planning research...", agent="researcher"))
        try:
            with self.tracer.span("researcher", query=state["query"]):
                plan = await self.researcher.plan(state["query"])
            return {"search_queries": plan.search_queries}
        except Exception as exc:  # noqa: BLE001 - node-level isolation by design
            logger.error("research node failed: %s", exc)
            writer(make_event("error", str(exc), agent="researcher"))
            return {
                "search_queries": [state["query"]],
                "errors": [*state["errors"], f"research: {exc}"],
            }

    async def retrieve(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        queries = state["search_queries"] or [state["query"]]
        writer(make_event(RETRIEVAL_STARTED, "Searching...", agent="retriever", queries=queries))
        try:
            documents = await self.retriever.retrieve(queries)
            writer(
                make_event(
                    RETRIEVAL_COMPLETED,
                    f"Found {len(documents)} candidate documents",
                    agent="retriever",
                    document_count=len(documents),
                )
            )
            return {"retrieved_documents": documents}
        except Exception as exc:  # noqa: BLE001
            logger.error("retrieve node failed: %s", exc)
            writer(make_event("error", str(exc), agent="retriever"))
            return {
                "retrieved_documents": [],
                "errors": [*state["errors"], f"retrieve: {exc}"],
            }

    async def grade(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        writer(make_event(AGENT_STARTED, "Evaluating evidence...", agent="grader"))
        try:
            with self.tracer.span("grader", document_count=len(state["retrieved_documents"])):
                graded = await self.grader.grade_and_filter(
                    state["query"], state["retrieved_documents"]
                )
            evidence = [
                RetrievedDocument(
                    document=g.document,
                    score=g.grade.score,
                    rank=rank,
                    retrieval_method="graded",
                )
                for rank, g in enumerate(graded, start=1)
            ]
            writer(
                make_event(
                    GRADING_COMPLETED,
                    f"{len(evidence)} of {len(state['retrieved_documents'])} documents relevant",
                    agent="grader",
                    relevant_count=len(evidence),
                    total_count=len(state["retrieved_documents"]),
                )
            )
            return {"graded_documents": graded, "evidence": evidence}
        except Exception as exc:  # noqa: BLE001
            logger.error("grade node failed: %s", exc)
            writer(make_event("error", str(exc), agent="grader"))
            return {
                "graded_documents": [],
                "evidence": [],
                "errors": [*state["errors"], f"grade: {exc}"],
            }

    async def synthesize(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        writer(make_event(GENERATION_STARTED, "Synthesizing...", agent="synthesizer"))
        try:
            result = None
            with self.tracer.span("synthesis", evidence_count=len(state["evidence"])):
                async for item in self.synthesizer.synthesize_streaming(
                    state["query"], state["evidence"]
                ):
                    if hasattr(item, "text"):  # LLMChunk
                        writer(make_event(TOKEN, item.text, agent="synthesizer"))
                    else:  # final SynthesisResult
                        result = item
            metadata = {**state["metadata"], "evidence_sufficient": result.evidence_sufficient}
            return {"answer": result.answer, "metadata": metadata}
        except Exception as exc:  # noqa: BLE001
            logger.error("synthesize node failed: %s", exc)
            writer(make_event("error", str(exc), agent="synthesizer"))
            return {"answer": "", "errors": [*state["errors"], f"synthesize: {exc}"]}

    async def cite(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        writer(make_event(AGENT_STARTED, "Verifying citations...", agent="citation"))
        try:
            with self.tracer.span("citation"):
                citations = self.citation.build_citations(state["answer"], state["evidence"])
                verified = self.citation.verify_citations(citations, state["evidence"])
            for c in verified:
                writer(make_event(CITATION, agent="citation", **c.model_dump()))
            return {"citations": verified}
        except Exception as exc:  # noqa: BLE001
            logger.error("cite node failed: %s", exc)
            writer(make_event("error", str(exc), agent="citation"))
            return {"citations": [], "errors": [*state["errors"], f"cite: {exc}"]}

    async def supervisor_review(self, state: ResearchState) -> dict:
        """No-op; the routing decision itself lives in `edges.py` so
        LangGraph's conditional-edge mechanism owns it. This node exists
        as an explicit place in the graph for that decision point.
        """
        return {}

    async def bump_retry(self, state: ResearchState) -> dict:
        return {"retry_count": state["retry_count"] + 1}

    async def rewrite_query(self, state: ResearchState) -> dict:
        """Rewrite search queries before retrying retrieval, per the
        Advanced Agentic RAG loop: Retrieve -> Grade -> insufficient? ->
        Rewrite query -> Retrieve.
        """
        writer = _safe_stream_writer()
        writer(make_event(AGENT_STARTED, "Refining search queries...", agent="researcher"))
        try:
            with self.tracer.span("researcher", rewrite=True):
                plan = await self.researcher.rewrite_queries(state["query"], state["search_queries"])
            return {"search_queries": plan.search_queries}
        except Exception as exc:  # noqa: BLE001
            logger.error("rewrite_query node failed: %s", exc)
            writer(make_event("error", str(exc), agent="researcher"))
            return {"errors": [*state["errors"], f"rewrite_query: {exc}"]}
