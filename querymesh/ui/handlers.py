"""UI-agnostic handlers for running a research query and streaming status.

Rule: "Keep UI logic separate from the graph." This module knows nothing
about Gradio — it consumes `querymesh.graph.run_research_workflow_streaming`
(Phase 22's structured `StreamEvent`s) and yields plain `UIUpdate` objects.
`gradio_app.py` is the only module that touches `gradio`.

Upgraded from Phase 21's original node-level streaming (which only ever
showed a fixed per-node status string, e.g. "Synthesizing...", with no
partial output) to real token-level streaming: the answer accumulates
live from `token` events as the LLM generates it, and citations appear
one at a time from `citation` events as `CitationAgent` verifies each
one — rather than the whole answer and every citation only appearing
once at the very end.
"""

from __future__ import annotations
from html import escape
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from langgraph.graph.state import CompiledStateGraph
from querymesh.observability import get_tracer
from querymesh.cache import get_response_cache
from querymesh.graph.state import create_initial_state
from querymesh.graph.workflow import run_research_workflow_streaming
from querymesh.graph.stream_events import CITATION, ERROR, TOKEN, WORKFLOW_COMPLETED

@dataclass
class UIUpdate:
    """One step of a streaming research run."""

    status: str
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    citations_html: str = ""
    retrieved_papers: list[dict] = field(default_factory=list)
    figure: str | None = None
    done: bool = False

def _citation_html(citations: list[dict], retrieved_papers: list[dict], ) -> str:
    """Build citation cards using the exact retrieved chunk text."""

    if not citations:
        return "<p>No citations generated.</p>"

    evidence_by_chunk: dict[str, dict] = {}

    for item in retrieved_papers:
        if not isinstance(item, dict):
            continue

        chunk_id = item.get("chunk_id")

        if not chunk_id:
            continue

        evidence_by_chunk[str(chunk_id)] = item

    blocks: list[str] = []

    for citation in citations:
        if not isinstance(citation, dict):
            continue

        citation_id = citation.get("citation_id")

        if citation_id is None:
            continue

        chunk_id = str(citation.get("chunk_id", ""))

        # -----------------------------------------------------
        # Find the exact retrieved chunk used by this citation.
        # -----------------------------------------------------
        evidence_item = evidence_by_chunk.get(chunk_id)

        # -----------------------------------------------------
        # Citation metadata
        # -----------------------------------------------------
        arxiv_id = str(citation.get("arxiv_id", "unknown", ))
        title = str(citation.get("title", "unknown title",))
        chunk_type = str(citation.get("chunk_type", "unknown",))
        page = citation.get("page")

        # -----------------------------------------------------
        # Get the actual supporting evidence.
        #
        # IMPORTANT:
        # retrieved_papers contains "text" directly.
        # -----------------------------------------------------
        evidence_text = ""

        if evidence_item:
            evidence_text = str(evidence_item.get("text", "",))

        if not evidence_text and evidence_item:
            # Table chunks may contain their content in
            # table_markdown instead of text.
            evidence_text = str(
                evidence_item.get("table_markdown", "",))

        if not evidence_text:
            evidence_text = ("The supporting evidence text is not available for this retrieved chunk.")

        # -----------------------------------------------------
        # Escape everything before inserting into HTML.
        # This prevents retrieved text from being interpreted
        # as HTML.
        # -----------------------------------------------------
        citation_id_html = escape(str(citation_id))
        arxiv_id_html = escape(arxiv_id)
        title_html = escape(title)
        chunk_id_html = escape(chunk_id)
        chunk_type_html = escape(chunk_type)
        evidence_text_html = escape(evidence_text)

        page_html = ""

        if page is not None:
            page_html = (
                f"""
                <div style="margin-top: 4px;">
                    <b>Page:</b> {escape(str(page))}
                </div>
                """
            )

        # -----------------------------------------------------
        # Citation card
        # -----------------------------------------------------
        blocks.append(
            f"""
            <div
                id="citation-{citation_id_html}"
                style="
                    border: 1px solid #777;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 12px 0;
                "
            >

                <div
                    style="
                        font-size: 1.05em;
                        margin-bottom: 10px;
                    "
                >
                    <b>
                        [{citation_id_html}] {title_html}
                    </b>
                </div>

                <div>
                    <b>arXiv:</b> {arxiv_id_html}
                </div>

                <div style="margin-top: 4px;">
                    <b>Chunk:</b>
                    <code>{chunk_id_html}</code>
                </div>

                <div style="margin-top: 4px;">
                    <b>Type:</b> {chunk_type_html}
                </div>

                {page_html}

                <div style="margin-top: 14px;">
                    <b>Supporting evidence:</b>
                </div>

                <blockquote
                    style="
                        margin: 8px 0 0 0;
                        padding: 12px;
                        border-left: 3px solid #888;
                        white-space: pre-wrap;
                        line-height: 1.5;
                    "
                >
                    {evidence_text_html}
                </blockquote>

                <div style="margin-top: 12px;">
                    <a href="#citation-{citation_id_html}">
                        Citation [{citation_id_html}]
                    </a>
                </div>

            </div>
            """
        )

    return "".join(blocks)

def _make_citations_clickable(answer: str, citations: list[dict], ) -> str:
    """Convert [1], [2], ... in the answer into clickable anchors.
    The answer itself remains Markdown. Only citation markers are replaced.
    """

    if not answer or not citations:
        return answer

    valid_ids = {
        str(citation.get("citation_id")) for citation in citations if citation.get("citation_id") is not None
    }

    if not valid_ids:
        return answer

    import re

    def replace(match: re.Match[str]) -> str:
        citation_id = match.group(1)

        if citation_id not in valid_ids:
            return match.group(0)

        return (
            f'<a href="#citation-{citation_id}">'
            f"[{citation_id}]"
            f"</a>"
        )

    return re.sub(r"\[(\d+)\]", replace, answer, )

async def run_query_streaming(compiled: CompiledStateGraph, query: str, max_retries: int = 2, ) -> AsyncIterator[UIUpdate]:
    """Run a query through the graph, yielding a `UIUpdate` for every
        structured stream event: the answer grows token by token as the LLM
        generates it, citations appear as each one is verified, and other
        events (`workflow_started`, `agent_started`, `retrieval_started`/
        `retrieval_completed`, `grading_completed`, `generation_started`)
        update the status line. The final `UIUpdate` has `done=True` and
        carries the complete answer, citations, and retrieved papers.
        """
    
    initial_state = create_initial_state(query, max_retries=max_retries, )
    tracer=get_tracer()
    cache = get_response_cache(tracer=tracer)
    answer_so_far = ""
    citations_so_far: list[dict] = []
    citations_html = "<p>No citations yet.</p>"
    retrieved_papers_so_far: list[dict] = []
    figure: str | None = None

    async for event in run_research_workflow_streaming(compiled=compiled, initial_state=initial_state, tracer=tracer, cache=cache):
        event_type = event.get("type")
            
        # ----------------------------------
        # Token from synthesizer
        # ----------------------------------
        if event_type == TOKEN:
            answer_so_far += event["message"]
            yield UIUpdate(status="Synthesizing...", answer=answer_so_far, citations=citations_so_far, citations_html=citations_html,
                retrieved_papers=retrieved_papers_so_far,
                figure=figure,)
        
        # ----------------------------------
        # Citation event
        # ----------------------------------
        elif event_type == CITATION:

            citations_so_far = [*citations_so_far, dict(event["data"])]
            citations_html = _citation_html(citations_so_far, retrieved_papers_so_far, )
            yield UIUpdate(status="Verifying citations...", answer=answer_so_far, citations=citations_so_far, citations_html=citations_html, retrieved_papers=retrieved_papers_so_far,
                figure=figure,)

        # ----------------------------------
        # Final workflow event
        # ----------------------------------
        elif event_type == WORKFLOW_COMPLETED:
            data = event["data"]
            
            final_answer = data.get("answer", answer_so_far,)
            final_citations = data.get("citations", citations_so_far, )
            
            if not isinstance(final_citations, list):
                final_citations = citations_so_far
                
            retrieved_data = data.get("retrieved_papers", [], )
            if not isinstance(retrieved_data, list):
                retrieved_data = []

            retrieved_papers_so_far = []

            for item in retrieved_data:
                if not isinstance(item, dict):
                    continue

                paper = dict(item)

                # Do not send embeddings to Gradio.
                paper.pop("embedding", None)

                retrieved_papers_so_far.append(paper)
        
            citations_html = _citation_html(citations_so_far, retrieved_papers_so_far, )
            clickable_answer = _make_citations_clickable(str(final_answer), citations_so_far, )
    
            yield UIUpdate(
                status="Done",
                answer=clickable_answer,                       #data.get("answer", answer_so_far),
                citations=citations_so_far,                            #data.get("citations", citations_so_far),
                citations_html=citations_html,
                retrieved_papers=data.get("retrieved_papers", []), 
                figure=figure,
                done=True,
            )
            
        # ----------------------------------
        # Error 
        # ----------------------------------
        elif event_type == ERROR:
            yield UIUpdate(status=f"Error: {event['message']}", answer=answer_so_far, done=True)
            return

        else:
            yield UIUpdate(status=event["message"] or event_type, answer=answer_so_far, citations=citations_so_far)



            # # Find first retrieved figure if we have retrieved documents
            # figure = None

            # for doc in retrieved_papers:
            #     if (doc.get("chunk_type") == "figure" and doc.get("image_path")):
            #         candidate = Path(doc["image_path"]).resolve()

            #         if candidate.exists():
            #             figure = str(candidate)
            #             break