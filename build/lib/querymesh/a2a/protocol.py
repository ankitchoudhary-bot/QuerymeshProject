"""A2A message protocol.

A minimal request/response/error envelope for agent-to-agent
communication — in-process today (Phase 17), but shaped so a future
network transport (HTTP/MCP) could carry the same envelope unchanged.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

MessageStatus = Literal["request", "response", "error"]


class AgentMessage(BaseModel):
    message_id: str
    sender: str
    recipient: str
    task: str
    payload: dict = Field(default_factory=dict)
    timestamp: datetime
    correlation_id: str
    status: MessageStatus
    hop_count: int = 0
    error: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_request(
    sender: str,
    recipient: str,
    task: str,
    payload: dict,
    correlation_id: str | None = None,
    hop_count: int = 0,
) -> AgentMessage:
    return AgentMessage(
        message_id=str(uuid.uuid4()),
        sender=sender,
        recipient=recipient,
        task=task,
        payload=payload,
        timestamp=_now(),
        correlation_id=correlation_id or str(uuid.uuid4()),
        status="request",
        hop_count=hop_count,
    )


def build_response(request: AgentMessage, payload: dict) -> AgentMessage:
    return AgentMessage(
        message_id=str(uuid.uuid4()),
        sender=request.recipient,
        recipient=request.sender,
        task=request.task,
        payload=payload,
        timestamp=_now(),
        correlation_id=request.correlation_id,
        status="response",
        hop_count=request.hop_count,
    )


def build_error(request: AgentMessage, error: str) -> AgentMessage:
    return AgentMessage(
        message_id=str(uuid.uuid4()),
        sender=request.recipient,
        recipient=request.sender,
        task=request.task,
        payload={},
        timestamp=_now(),
        correlation_id=request.correlation_id,
        status="error",
        hop_count=request.hop_count,
        error=error,
    )
