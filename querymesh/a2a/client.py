# """A2A client: sends messages to registered agent handlers.

# Transport is in-process today — a `recipient -> async handler` map — but
# the message envelope (`AgentMessage`) is transport-agnostic, so a future
# HTTP/MCP transport could replace the dispatch mechanism without changing
# callers' message-building code.

# Loop prevention: every message carries `hop_count`, incremented on each
# send. Exceeding `max_hops` raises `A2ALoopDetectedError` before ever
# invoking a handler — this bounds delegation chains deterministically,
# the same way Phase 13/14 bound the retry loop.
# """

# from __future__ import annotations

# import asyncio
# from typing import Awaitable, Callable

# from querymesh.a2a.errors import (
#     A2AAgentNotFoundError,
#     A2ALoopDetectedError,
#     A2ATimeoutError,
# )
# from querymesh.a2a.protocol import (
#     AgentMessage,
#     build_error,
#     build_request,
#     build_response,
# )
# from querymesh.observability.logging import get_logger
# from querymesh.observability.tracing import NoOpTracer, Tracer

# logger = get_logger(__name__)

# AgentHandler = Callable[[dict], Awaitable[dict]]

# _DEFAULT_TIMEOUT_SECONDS = 30.0
# _DEFAULT_MAX_RETRIES = 2
# _DEFAULT_MAX_HOPS = 10
# _RETRY_BASE_DELAY_SECONDS = 0.1


# class A2AClient:
#     """Dispatches A2A messages to registered in-process agent handlers."""

#     def __init__(
#         self,
#         timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
#         max_retries: int = _DEFAULT_MAX_RETRIES,
#         max_hops: int = _DEFAULT_MAX_HOPS,
#         tracer: Tracer | None = None,
#     ) -> None:
#         self.timeout_seconds = timeout_seconds
#         self.max_retries = max_retries
#         self.max_hops = max_hops
#         self.tracer = tracer or NoOpTracer()
#         self._handlers: dict[str, AgentHandler] = {}

#     def register_handler(self, agent_name: str, handler: AgentHandler) -> None:
#         self._handlers[agent_name] = handler

#     async def send(
#         self,
#         sender: str,
#         recipient: str,
#         task: str,
#         payload: dict,
#         correlation_id: str | None = None,
#         hop_count: int = 0,
#     ) -> AgentMessage:
#         """Send a request and return the response (or error) envelope.

#         Never raises for a recipient-side failure or timeout — those come
#         back as an error-status `AgentMessage` so callers can inspect
#         `response.status`/`response.error` uniformly. Only genuinely
#         unrecoverable client-side conditions (unknown recipient, loop
#         detected) raise directly, since there's no request to attach an
#         error response to yet.
#         """
#         if hop_count >= self.max_hops:
#             raise A2ALoopDetectedError(f"Message from '{sender}' to '{recipient}' exceeded max_hops={self.max_hops}")

#         handler = self._handlers.get(recipient)
#         if handler is None:
#             raise A2AAgentNotFoundError(f"No handler registered for recipient '{recipient}'")

#         request = build_request(
#             sender=sender,
#             recipient=recipient,
#             task=task,
#             payload=payload,
#             correlation_id=correlation_id,
#             hop_count=hop_count,
#         )

#         last_exc: Exception | None = None
        
#         for attempt in range(self.max_retries + 1):
            
#             try:
#                 result = await asyncio.wait_for(handler(request.payload), timeout=self.timeout_seconds)
#                 return build_response(request, result)
            
#             except TimeoutError:
#                 last_exc = A2ATimeoutError(f"'{recipient}' did not respond within {self.timeout_seconds}s")
#                 logger.warning(
#                     "A2A call to '%s' timed out (attempt %d/%d)",
#                     recipient,
#                     attempt + 1,
#                     self.max_retries + 1,
#                 )
#             except Exception as exc:
#                 last_exc = exc
#                 logger.warning(
#                     "A2A call to '%s' failed (attempt %d/%d): %s",
#                     recipient,
#                     attempt + 1,
#                     self.max_retries + 1,
#                     exc,
#                 )
#             if attempt < self.max_retries:
#                 await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))

#         return build_error(request, str(last_exc))





"""A2A client: sends messages to registered agent handlers.

Two transports, same interface: `register_handler()` for an in-process
`recipient -> async handler` map (single-process deployment), and
`register_remote_agent()` for a real, separately-running agent reachable
over HTTP/MCP (`querymesh.a2a.http_transport.MCPHttpTransport`). Both are
just `AgentHandler` callables under the hood, so `send()`'s retry/timeout/
loop-prevention logic — and every caller of `send()` — is identical
either way; only how a recipient was registered differs.

Loop prevention: every message carries `hop_count`, incremented on each
send. Exceeding `max_hops` raises `A2ALoopDetectedError` before ever
invoking a handler — this bounds delegation chains deterministically,
the same way Phase 13/14 bound the retry loop.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from querymesh.a2a.errors import (
    A2AAgentNotFoundError,
    A2ALoopDetectedError,
    A2ATimeoutError,
)
from querymesh.a2a.http_transport import MCPHttpTransport
from querymesh.a2a.protocol import (
    AgentMessage,
    build_error,
    build_request,
    build_response,
)
from querymesh.observability.logging import get_logger
from querymesh.observability.tracing import NoOpTracer, Tracer

logger = get_logger(__name__)

AgentHandler = Callable[[dict], Awaitable[dict]]

_DEFAULT_TIMEOUT_SECONDS = 30.0
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_MAX_HOPS = 10
_RETRY_BASE_DELAY_SECONDS = 0.1


class A2AClient:
    """Dispatches A2A messages to registered agent handlers — in-process
    or over a real HTTP/MCP transport, per recipient (see
    `register_handler`/`register_remote_agent`).
    """

    def __init__(
        self,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        max_hops: int = _DEFAULT_MAX_HOPS,
        tracer: Tracer | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_hops = max_hops
        self.tracer = tracer or NoOpTracer()
        self._handlers: dict[str, AgentHandler] = {}

    def register_handler(self, agent_name: str, handler: AgentHandler) -> None:
        self._handlers[agent_name] = handler

    def register_remote_agent(
        self,
        agent_name: str,
        url: str,
        tool_name: str,
        timeout_seconds: float | None = None,
    ) -> None:
        """Register a recipient backed by a real, separately-running MCP
        server rather than an in-process handler.

        `url` is the remote agent's FastMCP HTTP endpoint (e.g.
        `"http://researcher-agent:8765/mcp"`); `tool_name` is the tool it
        exposes for this recipient (`search_papers`, `retrieve_documents`,
        `grade_documents`, `generate_answer`, or `verify_citations` — see
        `querymesh.mcp.servers`). Once registered, `send(recipient=agent_name,
        ...)` behaves identically to an in-process handler — same retries,
        timeout, and loop prevention — the only difference is the call
        goes out over the network.
        """
        self.register_handler(
            agent_name,
            MCPHttpTransport(url=url, tool_name=tool_name, timeout_seconds=timeout_seconds),
        )

    async def send(
        self,
        sender: str,
        recipient: str,
        task: str,
        payload: dict,
        correlation_id: str | None = None,
        hop_count: int = 0,
    ) -> AgentMessage:
        """Send a request and return the response (or error) envelope.

        Never raises for a recipient-side failure or timeout — those come
        back as an error-status `AgentMessage` so callers can inspect
        `response.status`/`response.error` uniformly. Only genuinely
        unrecoverable client-side conditions (unknown recipient, loop
        detected) raise directly, since there's no request to attach an
        error response to yet.
        """
        if hop_count >= self.max_hops:
            raise A2ALoopDetectedError(
                f"Message from '{sender}' to '{recipient}' exceeded max_hops={self.max_hops}"
            )

        handler = self._handlers.get(recipient)
        if handler is None:
            raise A2AAgentNotFoundError(f"No handler registered for recipient '{recipient}'")

        request = build_request(
            sender=sender,
            recipient=recipient,
            task=task,
            payload=payload,
            correlation_id=correlation_id,
            hop_count=hop_count,
        )

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await asyncio.wait_for(handler(request.payload), timeout=self.timeout_seconds)
                return build_response(request, result)
            except TimeoutError:
                last_exc = A2ATimeoutError(
                    f"'{recipient}' did not respond within {self.timeout_seconds}s"
                )
                logger.warning(
                    "A2A call to '%s' timed out (attempt %d/%d)",
                    recipient,
                    attempt + 1,
                    self.max_retries + 1,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "A2A call to '%s' failed (attempt %d/%d): %s",
                    recipient,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
            if attempt < self.max_retries:
                await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))

        return build_error(request, str(last_exc))
