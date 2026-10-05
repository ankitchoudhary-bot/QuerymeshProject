"""Real HTTP/MCP transport for A2A calls to separately-running agent
processes.

Phase 17 originally shipped A2A with an in-process `recipient -> handler`
map — correct for a single-process deployment, but not an actual network
transport. This module adds the real thing: `MCPHttpTransport` wraps a
`fastmcp.Client` connection to a remote agent's MCP server (Phase 16) and
exposes it as an `AgentHandler` — the same `Callable[[dict], Awaitable[dict]]`
shape `A2AClient.register_handler()` already accepts. That's deliberate:
`A2AClient.send()`'s retry/timeout/loop-prevention logic doesn't need to
know or care whether a handler runs in-process or over the network — the
transport is swapped in as just another handler, not a parallel code path.

Each QueryMesh agent (Phases 11/16) already runs as its own FastMCP
server exposing one tool (`search_papers`, `retrieve_documents`,
`grade_documents`, `generate_answer`, `verify_citations`) — those servers
*are* "separately-running agent processes" once started with
`server.run(transport="http", ...)`. This module is the client side that
talks to them.
"""

from __future__ import annotations

from fastmcp import Client

from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


class MCPHttpTransport:
    """Calls one tool on a remote FastMCP server over HTTP, presented as
    an `AgentHandler` (`Callable[[dict], Awaitable[dict]]`).

    A new `Client` connection is opened per call rather than held open
    across calls — simpler and more robust for a request/response
    pattern than managing a long-lived connection's lifecycle across
    however many `A2AClient.send()` calls happen to occur, at the cost of
    one extra connection handshake per call. That's a reasonable
    trade-off for the request volumes A2A delegation involves; a
    connection-pooling version is a natural follow-up if this becomes a
    hot path.

    List-returning tools (`retrieve_documents`, `grade_documents`,
    `verify_citations` all return `list[dict]`) are wrapped as
    `{"items": [...]}` before returning, since `AgentMessage.payload` is
    typed as `dict` — a list can't go there directly. Dict-returning
    tools (`search_papers`, `generate_answer`) pass through unchanged.
    """

    def __init__(self, url: str, tool_name: str, timeout_seconds: float | None = None) -> None:
        if not url:
            raise ValueError("MCPHttpTransport requires a non-empty url")
        if not tool_name:
            raise ValueError("MCPHttpTransport requires a non-empty tool_name")
        self.url = url
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds

    async def __call__(self, payload: dict) -> dict:
        logger.debug("MCPHttpTransport calling %s on %s", self.tool_name, self.url)
        async with Client(self.url, timeout=self.timeout_seconds) as client:
            result = await client.call_tool(self.tool_name, payload)

        data = result.data
        if isinstance(data, list):
            return {"items": data}
        if isinstance(data, dict):
            return data
        # Defensive: a tool returning a bare scalar would violate the MCP
        # servers' own declared return types (all are `dict` or
        # `list[dict]`, Phase 16), but don't let an unexpected shape
        # silently vanish — surface it.
        return {"result": data}
