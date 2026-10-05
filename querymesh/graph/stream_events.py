"""Structured streaming event model.

Every event the graph emits (via LangGraph's custom stream mode, plus a
thin wrapper for workflow-level start/end) takes this shape, so UI layers
consume one uniform event type regardless of what produced it.
"""

from __future__ import annotations

from typing import Any, TypedDict


class StreamEvent(TypedDict):
    type: str
    agent: str | None
    message: str
    data: dict[str, Any]


# Event types, exactly as specced.
WORKFLOW_STARTED = "workflow_started"
AGENT_STARTED = "agent_started"
RETRIEVAL_STARTED = "retrieval_started"
RETRIEVAL_COMPLETED = "retrieval_completed"
GRADING_COMPLETED = "grading_completed"
GENERATION_STARTED = "generation_started"
TOKEN = "token"
CITATION = "citation"
WORKFLOW_COMPLETED = "workflow_completed"
ERROR = "error"


def make_event(event_type: str, message: str = "", agent: str | None = None, **data: Any) -> StreamEvent:
    return StreamEvent(type=event_type, agent=agent, message=message, data=data)
