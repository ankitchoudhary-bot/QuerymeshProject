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

from querymesh.a2a.client import A2AClient
from querymesh.agents.citation import CitationAgent
from querymesh.agents.models import Citation
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
from querymesh.cache.response_cache import ResponseCache
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


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
    citation_llm: LLM | None = None,
    tracer: Tracer | None = None,
    a2a_client: A2AClient | None = None,
) -> CompiledStateGraph:
    """Build agent instances and compile the full research graph.
    
    `retrieval_service` is always required to construct `RetrieverAgent`
    (its constructor needs one regardless), but note that when
    `a2a_client` is set, `GraphNodes.retrieve()` dispatches to the
    `retriever` MCP recipient via A2A instead of calling
    `RetrieverAgent.retrieve()` directly — `retrieval_service` then only
    serves as the harmless fallback for the (a2a_client is None) branch.
    In an A2A deployment, the actual retrieval logic runs inside whatever
    process is serving the `retrieval-agent` MCP server (built with
    `querymesh.mcp.servers.build_retrieval_server`), not in this process.

    `citation_llm` (like `grading_llm`) defaults to `llm` — `CitationAgent`
    needs one to map answer sentences to their supporting evidence chunks.

    Pass `a2a_client` with recipients already registered (via
    `a2a_client.register_handler()` for in-process handlers or
    `a2a_client.register_remote_agent()` for real, separately-running MCP
    servers — see `querymesh.a2a.http_transport.MCPHttpTransport`) to make
    every node dispatch through it instead of calling its agent directly.
    """
    nodes = GraphNodes(
        researcher=ResearcherAgent(llm),
        retriever=RetrieverAgent(retrieval_service),
        grader=GraderAgent(grading_llm or llm),
        synthesizer=SynthesizerAgent(llm),
        citation=CitationAgent(citation_llm or llm),
        tracer=tracer or NoOpTracer(),
        a2a_client=a2a_client
    )
    return build_workflow(nodes).compile()

_DEFAULT_RECURSION_LIMIT = 50

def _dump(item: Any) -> dict:
    return item.model_dump() if hasattr(item, "model_dump") else item

def _cache_metadata(result: dict[str, Any]) -> dict[str, Any]:
    """What's worth remembering about a fresh graph result alongside its
    cached answer/citations — just the retrieved-document count today,
    since that's the one thing a cache hit can't otherwise report (the
    cache never stores full `retrieved_documents`/`evidence` detail, only
    the final synthesized answer and citations).
    """
    return {"retrieved_document_count": len(result.get("retrieved_documents", []))}


def _result_from_cache(initial_state: ResearchState, cached: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a graph-result-shaped dict from a cache hit.

    Shaped to match what a real `compiled.ainvoke()` result looks like,
    so callers (the API layer, `run_research_workflow_streaming`, tests)
    don't need to special-case a cache hit — `result["answer"]`,
    `result["citations"]` (real `Citation` objects, not dicts, matching
    what the graph itself produces), etc. all work the same way.

    `retrieved_documents`/`graded_documents`/`evidence` come back empty —
    a documented trade-off, not an oversight: the cache only ever stored
    the final answer and citations (Phase 19's `ResponseCache.set()`
    signature doesn't carry per-document detail), so there is nothing
    to reconstruct there. `metadata["cache_hit"] = True` lets a caller
    that cares distinguish this from a real run; the original
    `retrieved_document_count` (from when the answer was first computed)
    is preserved in metadata for exactly that reason.
    """
    metadata = {**initial_state["metadata"], **(cached.get("metadata") or {}), "cache_hit": True}

    return {
        "query": initial_state["query"],
        "search_queries": initial_state["search_queries"],
        "retrieved_documents": [],
        "graded_documents": [],
        "evidence": [],
        "answer": cached.get("answer", ""),
        "citations": [Citation(**c) for c in cached.get("citations", [])],
        "retry_count": initial_state["retry_count"],
        "max_retries": initial_state["max_retries"],
        "errors": [],
        "metadata": metadata,
    }
    
async def _cache_get_safe(cache: ResponseCache, query: str) -> dict[str, Any] | None:
    """`cache.get()` failing (e.g. Redis unreachable) must never crash the
    request — caching is an optional performance optimization, not a
    dependency of the actual research pipeline. Treat any failure as a
    cache miss and let the real graph run instead. Mirrors the same
    "never let an optional component break the request" philosophy
    `api/dependencies.py` already applies at startup — this closes the
    gap for the identical failure happening later, at request time.
    """
    try:
        return await cache.get(query)
    except Exception as exc:  # noqa: BLE001 - any cache backend failure degrades to a miss
        logger.warning("Cache lookup failed, proceeding without it: %s", exc)
        return None


async def _cache_set_safe(cache: ResponseCache, query: str, **kwargs: Any) -> None:
    """Symmetric with `_cache_get_safe`: failing to *write* the cache
    must not fail an otherwise-successful request either.
    """
    try:
        await cache.set(query, **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache write failed, continuing without caching this result: %s", exc)



   
async def run_research_workflow(
    compiled: CompiledStateGraph,
    initial_state: ResearchState,
    tracer: Tracer | None = None,
    recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
    cache: ResponseCache | None = None,
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

    `cache`, if given, closes the loop Phase 19 left open: checked before
    the graph runs at all (a hit skips every LLM/retrieval call entirely
    — the "cache" span still fires via `ResponseCache.get()` itself, with
    `cache_hit=True`/`False`), and populated after a successful run so the
    next identical query is fast. Per `ResponseCache.set()`'s own rule, a
    run that produced no answer (empty string) is never cached. A cache
    backend failure at request time (e.g. Redis down) degrades to running
    the real graph rather than failing the request.
    """
    query=initial_state["query"]
    tracer = tracer or NoOpTracer()
    
    if cache is not None:
            cached = await _cache_get_safe(cache, query)
            if cached is not None:
                return _result_from_cache(initial_state, cached)
    
    with tracer.span("research_request", query=query):
        result = await compiled.ainvoke(initial_state, config={"recursion_limit": recursion_limit})
    
        if cache is not None:
            await _cache_set_safe(
                cache,
                query,
                answer=result.get("answer", ""),
                citations=[_dump(c) for c in result.get("citations", [])],
                metadata=_cache_metadata(result),
            )
    
        return result
    
    
async def run_research_workflow_streaming(
    compiled: CompiledStateGraph,
    initial_state: ResearchState,
    tracer: Tracer | None = None,
    recursion_limit: int = _DEFAULT_RECURSION_LIMIT,
    cache: ResponseCache | None = None,
) -> AsyncIterator[StreamEvent]:
    """Run a compiled graph, yielding structured `StreamEvent`s as the
    workflow progresses: `workflow_started`, per-node `agent_started`/
    `retrieval_started`/`retrieval_completed`/`grading_completed`/
    `generation_started`/`token`/`citation` events (emitted by the nodes
    themselves via LangGraph's custom stream mode), and finally
    `workflow_completed` (or `error` if the run raises).
    
    `cache`, if given and hit, short-circuits straight from
    `workflow_started` to a `workflow_completed` carrying
    `data["cache_hit"] = True` and the cached answer/citations — no
    per-node events fire, since no node runs. A cache miss streams
    normally and populates the cache from the real result afterward,
    exactly like `run_research_workflow`. A cache backend failure (get or
    set) degrades to running/streaming the real graph rather than
    crashing the stream, via the same `_cache_get_safe`/`_cache_set_safe`
    wrappers `run_research_workflow` uses.
    """
    tracer = tracer or NoOpTracer()
    query = initial_state["query"]
    
    yield make_event(WORKFLOW_STARTED, "Starting research request", query=query, )
    
    if cache is not None:
        cached = await _cache_get_safe(cache, query)
        if cached is not None:
            logger.info("Response cache HIT for query=%r", query)
            cached_citations = cached.get("citations", [])
            yield make_event(
                WORKFLOW_COMPLETED,
                "Answer served from cache",
                answer=cached.get("answer", ""),
                citation_count=len(cached_citations),
                retry_count=0,
                citations=cached_citations,
                retrieved_papers=[],
                cache_hit=True,
            )
            return
            
        logger.info("Response cache MISS for query=%r", query)

    try:
        with tracer.span("research_request", query=query):
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
    except Exception as exc:
        logger.exception("Research workflow streaming failed for query=%r", query, )
        yield make_event("error", str(exc))
        return
    
    citations = [_dump(c) for c in final_state.get("citations", [])]
    retrieved_papers = [_dump(e).get("document", _dump(e)) for e in final_state.get("evidence", [])]
    
    
    if cache is not None:
        await _cache_set_safe(
            cache,
            query,
            answer=final_state.get("answer", ""),
            citations=citations,
            metadata=_cache_metadata(final_state),
        )
        
        logger.info("Response cached for query=%r", query, )
    
    yield make_event(
    WORKFLOW_COMPLETED,
    "Research request complete",
    answer=final_state.get("answer", ""),
    citation_count=len(citations),
    retry_count=final_state.get("retry_count", 0),
    citations=citations,
    retrieved_papers=retrieved_papers,
    cache_hit=False,
)