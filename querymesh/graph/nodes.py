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

from querymesh.a2a.client import A2AClient
from querymesh.agents.models import (
    Citation,
    DocumentGradeResult,
    GradedDocument,
    ResearchPlan,
    SynthesisResult,
)
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
    
def _retrieval_document_for_ui(rd: RetrievedDocument) -> dict:
    """Return a UI-safe representation of a retrieved document."""
    doc = dict(rd.document)

    # Never send the huge embedding vector to the UI/event stream.
    doc.pop("embedding", None)

    return {
        **doc,
        "score": rd.score,
        "rank": rd.rank,
        "retrieval_method": rd.retrieval_method,
    }

# A2A recipient names match `querymesh.a2a.registry_builder.build_default_registry`'s
# naming exactly, so the same registry/client wiring works whether a
# recipient is backed by an in-process handler or a real remote MCP
# server (querymesh.a2a.http_transport.MCPHttpTransport).
_RESEARCHER_RECIPIENT = "researcher"
_RETRIEVER_RECIPIENT = "retriever"
_GRADER_RECIPIENT = "grader"
_SYNTHESIZER_RECIPIENT = "synthesizer"
_CITATION_RECIPIENT = "citation"

# Tool names match querymesh.mcp.servers exactly (Phase 16).
_SEARCH_PAPERS_TOOL = "search_papers"
_REWRITE_SEARCH_QUERIES_TOOL = "rewrite_search_queries"
_RETRIEVE_DOCUMENTS_TOOL = "retrieve_documents"
_GRADE_DOCUMENTS_TOOL = "grade_documents"
_GENERATE_ANSWER_TOOL = "generate_answer"
_VERIFY_CITATIONS_TOOL = "verify_citations"
  
@dataclass
class GraphNodes:
    researcher: ResearcherAgent
    retriever: RetrieverAgent
    grader: GraderAgent
    synthesizer: SynthesizerAgent
    citation: CitationAgent
    tracer: Tracer = field(default_factory=NoOpTracer)
    a2a_client: A2AClient | None = None
    
    """When set, every node (research/rewrite_query, retrieve, grade,
        synthesize, cite) dispatches to its agent via `A2AClient.send()`
        (task name = the recipient's MCP tool name, see the `_*_TOOL`
        constants above) instead of calling the agent object directly
        in-process. The recipient names the client's handlers must be
        registered under are `_RESEARCHER_RECIPIENT`/etc. above — either as
        in-process handlers (`register_handler`) or real remote agents
        (`register_remote_agent`, backed by `MCPHttpTransport`). Leave this
        `None` (the default) for the original, zero-network in-process
        behavior every prior phase's tests rely on.
    
        One real capability trade-off when using A2A dispatch: `synthesize()`
        loses token-level streaming — `generate_answer` is a single
        request/response MCP tool call, not a stream, so no `TOKEN` events
        fire on this path. That's an honest architectural trade-off (network
        request/response vs. an in-process async generator), not something
        worth faking with synthetic per-word chunks of an already-complete
        answer.
        """
    
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
                if self.a2a_client is not None:
                    plan = await self._research_via_a2a(state["query"])
                else:
                    plan = await self.researcher.plan(state["query"])
            return {"search_queries": plan.search_queries}
        except Exception as exc:  # noqa: BLE001 - node-level isolation by design
            logger.error("research node failed: %s", exc)
            writer(make_event("error", str(exc), agent="researcher"))
            return {
                "search_queries": [state["query"]],
                "errors": [*state["errors"], f"research: {exc}"],
            }
            
    async def _research_via_a2a(self, query: str) -> ResearchPlan:
            response = await self.a2a_client.send(
                sender="supervisor",
                recipient=_RESEARCHER_RECIPIENT,
                task=_SEARCH_PAPERS_TOOL,
                payload={"query": query},
            )
            if response.status == "error":
                raise RuntimeError(f"researcher agent returned an error: {response.error}")
            queries = response.payload.get("search_queries") or [query]
            return ResearchPlan(search_queries=queries, reasoning=response.payload.get("reasoning"))

    async def retrieve(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        queries = state["search_queries"] or [state["query"]]
        writer(make_event(RETRIEVAL_STARTED, "Searching...", agent="retriever", queries=queries))
        try:
            if self.a2a_client is not None:
                documents = await self._retrieve_via_a2a(queries)
            else:  
                documents = await self.retriever.retrieve(queries)
            ui_documents = [_retrieval_document_for_ui(rd) for rd in documents]
            writer(
                make_event(
                    RETRIEVAL_COMPLETED,
                    f"Found {len(documents)} candidate documents",
                    agent="retriever",
                    document_count=len(documents),
                    documents=ui_documents,
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
            
    async def _retrieve_via_a2a(self, queries: list[str]) -> list[RetrievedDocument]:
            response = await self.a2a_client.send(
                sender="supervisor",
                recipient=_RETRIEVER_RECIPIENT,
                task=_RETRIEVE_DOCUMENTS_TOOL,
                payload={"queries": queries, "top_k": 10},
            )
            if response.status == "error":
                raise RuntimeError(f"retriever agent returned an error: {response.error}")
            return [RetrievedDocument(**item) for item in response.payload.get("items", [])]


    async def grade(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        writer(make_event(AGENT_STARTED, "Evaluating evidence...", agent="grader"))
        try:
            with self.tracer.span("grader", document_count=len(state["retrieved_documents"])):
                
                if self.a2a_client is not None:
                    graded = await self._grade_via_a2a(state["query"], state["retrieved_documents"])
                # graded = await self.grader.grade_and_filter(
                #     state["query"], state["retrieved_documents"]
                # )
                else:
                    graded = await self.grader.grade_all(state["query"], state["retrieved_documents"])
            evidence = [
                RetrievedDocument(
                    document=g.document,
                    score=g.grade.score,
                    rank=rank,
                    retrieval_method="graded",
                )
                for rank, g in enumerate(graded, start=1) if g.grade.relevant and g.grade.score >= self.grader.relevance_threshold
            ]
            
            # Collect reasons from documents that were rejected.
            grading_reasons = [g.grade.reason for g in graded
                if (not g.grade.relevant or g.grade.score < self.grader.relevance_threshold)
                and g.grade.reason.strip()
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
            return {"graded_documents": graded, "evidence": evidence, "grading_reasons": grading_reasons, }
        except Exception as exc:  # noqa: BLE001
            logger.error("grade node failed: %s", exc)
            writer(make_event("error", str(exc), agent="grader"))
            return {
                "graded_documents": [],
                "evidence": [],
                "grading_reasons": [],
                "errors": [*state["errors"], f"grade: {exc}"],
            }
            
    async def _grade_via_a2a(
            self, query: str, documents: list[RetrievedDocument]
        ) -> list[GradedDocument]:
            response = await self.a2a_client.send(
                sender="supervisor",
                recipient=_GRADER_RECIPIENT,
                task=_GRADE_DOCUMENTS_TOOL,
                payload={"query": query, "documents": [d.model_dump() for d in documents]},
            )
            if response.status == "error":
                raise RuntimeError(f"grader agent returned an error: {response.error}")
            items = response.payload.get("items", [])
            return [GradedDocument(**item) for item in items]
    
    async def synthesize(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        writer(make_event(GENERATION_STARTED, "Synthesizing...", agent="synthesizer"))
        try:
            with self.tracer.span("synthesis", evidence_count=len(state["evidence"])):
                if self.a2a_client is not None:
                    # A single MCP request/response call, not a stream — no
                    # TOKEN events fire on this path. See the `a2a_client`
                    # field docstring for why that's an honest trade-off
                    # rather than something worth faking.
                    result = await self._synthesize_via_a2a(state["query"], state["evidence"])
                else:
                    result =None
                    async for item in self.synthesizer.synthesize_streaming(state["query"], state["evidence"]):
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

    async def _synthesize_via_a2a(
            self, query: str, evidence: list[RetrievedDocument]
        ) -> SynthesisResult:
            response = await self.a2a_client.send(
                sender="supervisor",
                recipient=_SYNTHESIZER_RECIPIENT,
                task=_GENERATE_ANSWER_TOOL,
                payload={"query": query, "evidence": [e.model_dump() for e in evidence]},
            )
            if response.status == "error":
                raise RuntimeError(f"synthesizer agent returned an error: {response.error}")
            return SynthesisResult(**response.payload)
    
    
    async def cite(self, state: ResearchState) -> dict:
        writer = _safe_stream_writer()
        writer(make_event(AGENT_STARTED, "Verifying citations...", agent="citation"))
        try:
            with self.tracer.span("citation"):
                if self.a2a_client is not None:
                    verified = await self._cite_via_a2a(state["query"], state["answer"], state["evidence"])
                    # A2A currently returns only citations, so preserve
                    # the original answer in this path.
                    answer = state["answer"]
                else:
                    
                    citations = await self.citation.build_citations(state["answer"], state["evidence"], query=state["query"])   #later added query=state["query"]
                    answer= citations.answer
                    verified = self.citation.verify_citations(citations.citations, state["evidence"])
                    
            for c in verified:
                writer(make_event(CITATION, agent="citation", **c.model_dump()))
            return {"answer": answer, "citations": verified}
        
        except Exception as exc:  # noqa: BLE001
            logger.error("cite node failed: %s", exc)
            writer(make_event("error", str(exc), agent="citation"))
            return {"citations": [], "errors": [*state["errors"], f"cite: {exc}"]}
        
    async def _cite_via_a2a(
            self, query: str, answer: str, evidence: list[RetrievedDocument]
        ) -> list[Citation]:
            response = await self.a2a_client.send(
                sender="supervisor",
                recipient=_CITATION_RECIPIENT,
                task=_VERIFY_CITATIONS_TOOL,
                payload={"answer": answer, "evidence": [e.model_dump() for e in evidence], "query": query},
            )
            if response.status == "error":
                raise RuntimeError(f"citation agent returned an error: {response.error}")
            items = response.payload.get("items", [])
            return [Citation(**item) for item in items]
    

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
                if self.a2a_client is not None:
                    plan = await self._rewrite_query_via_a2a(state["query"], state["search_queries"])
                else:
                    grading_reason = "\n".join(f"- {reason}" for reason in state.get("grading_reasons", []))
                    plan = await self.researcher.rewrite_queries(state["query"], state["search_queries"], grading_reason=grading_reason or None)
                    
            return {"search_queries": plan.search_queries}
        
        except Exception as exc:  # noqa: BLE001
            logger.error("rewrite_query node failed: %s", exc)
            writer(make_event("error", str(exc), agent="researcher"))
            return {"errors": [*state["errors"], f"rewrite_query: {exc}"]}


    async def _rewrite_query_via_a2a(self, query: str, previous_queries: list[str]) -> ResearchPlan:
        response = await self.a2a_client.send(
            sender="supervisor",
            recipient=_RESEARCHER_RECIPIENT,
            task=_REWRITE_SEARCH_QUERIES_TOOL,
            payload={"query": query, "previous_queries": previous_queries},
        )
        if response.status == "error":
            raise RuntimeError(f"researcher agent returned an error: {response.error}")
        queries = response.payload.get("search_queries") or previous_queries or [query]
        return ResearchPlan(search_queries=queries, reasoning=response.payload.get("reasoning"))


