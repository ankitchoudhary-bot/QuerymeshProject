"""UI-agnostic handlers for running a research query and streaming status.

Rule: "Keep UI logic separate from the graph." This module knows nothing
about Gradio — it consumes the compiled LangGraph workflow via LangGraph's
own `astream(..., stream_mode="updates")` and yields plain `UIUpdate`
objects. `gradio_app.py` is the only module that touches `gradio`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from querymesh.graph.state import create_initial_state

_STATUS_MESSAGES: dict[str, str] = {
    "supervisor_start": "Starting research...",
    "research": "Planning research...",
    "retrieve": "Searching...",
    "grade": "Evaluating evidence...",
    "synthesize": "Synthesizing...",
    "cite": "Verifying citations...",
    "supervisor_review": "Reviewing evidence sufficiency...",
    "bump_retry": "Retrying with a broader search...",
    "rewrite_query": "Refining search queries...",
}


def _dump(item: Any) -> dict:
    return item.model_dump() if hasattr(item, "model_dump") else item


@dataclass
class UIUpdate:
    """One step of a streaming research run."""

    status: str
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    retrieved_papers: list[dict] = field(default_factory=list)
    done: bool = False


async def run_query_streaming(
    compiled: CompiledStateGraph, query: str, max_retries: int = 2
) -> AsyncIterator[UIUpdate]:
    """Run a query through the graph, yielding a `UIUpdate` after each node
    completes, then a final `UIUpdate` with `done=True` carrying the answer,
    citations, and retrieved papers.
    """
    initial_state = create_initial_state(query, max_retries=max_retries)
    accumulated: dict[str, Any] = dict(initial_state)

    async for event in compiled.astream(initial_state, stream_mode="updates"):
        for node_name, update in event.items():
            if update:
                accumulated.update(update)
            yield UIUpdate(status=_STATUS_MESSAGES.get(node_name, node_name))

    citations = [_dump(c) for c in accumulated.get("citations", [])]
    retrieved_papers = [_dump(e).get("document", _dump(e)) for e in accumulated.get("evidence", [])]

    yield UIUpdate(
        status="Done",
        answer=accumulated.get("answer", ""),
        citations=citations,
        retrieved_papers=retrieved_papers,
        done=True,
    )
