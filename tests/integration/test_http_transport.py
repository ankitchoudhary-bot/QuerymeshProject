"""Tests for `MCPHttpTransport` against a genuinely running FastMCP HTTP
server — real TCP sockets, real HTTP requests over loopback, not
in-memory/mocked transport. Each test starts a real `uvicorn`-backed
FastMCP server (via `FastMCP.run_async(transport="http", ...)`) on its
own port as a background asyncio task, then has `MCPHttpTransport`
connect to it exactly as it would connect to a separately-running agent
process in production — the only difference from a real separate OS
process is that this one happens to share the test's process, not the
network path, the HTTP framing, or any of the code under test.
"""

from __future__ import annotations

import asyncio
import socket

import pytest
from fastmcp import FastMCP

from querymesh.a2a.http_transport import MCPHttpTransport


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RunningServer:
    """Starts a real FastMCP HTTP server in the background and tears it
    down cleanly — a small harness so each test doesn't repeat the
    start/wait/cancel dance.
    """

    def __init__(self, mcp: FastMCP) -> None:
        self.mcp = mcp
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> "_RunningServer":
        self._task = asyncio.create_task(
            self.mcp.run_async(
                transport="http", host="127.0.0.1", port=self.port, show_banner=False
            )
        )
        # Real startup: give uvicorn's real socket bind/accept loop time to
        # come up before the first request. A fixed sleep is crude but
        # reliable here — the alternative (polling the port) adds
        # complexity for a test-only concern.
        await asyncio.sleep(1.0)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        assert self._task is not None
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass


def _build_dict_server() -> FastMCP:
    mcp = FastMCP(name="test-dict-agent")

    @mcp.tool
    async def search_papers(query: str) -> dict:
        return {"search_queries": [f"{query} refined"], "reasoning": "test"}

    return mcp


def _build_list_server() -> FastMCP:
    mcp = FastMCP(name="test-list-agent")

    @mcp.tool
    async def retrieve_documents(queries: list[str], top_k: int = 10) -> list[dict]:
        return [{"chunk_id": f"{q}:chunk:0000", "score": 1.0} for q in queries]

    return mcp


def _build_failing_server() -> FastMCP:
    mcp = FastMCP(name="test-failing-agent")

    @mcp.tool
    async def generate_answer(query: str, evidence: list[dict]) -> dict:
        raise RuntimeError("synthesis backend unavailable")

    return mcp


# --- Real HTTP round trip: dict-returning tool -----------------------------


@pytest.mark.integration
async def test_transport_calls_real_dict_returning_tool_over_http() -> None:
    async with _RunningServer(_build_dict_server()) as server:
        transport = MCPHttpTransport(url=server.url, tool_name="search_papers")
        result = await transport({"query": "attention mechanisms"})

    assert result == {"search_queries": ["attention mechanisms refined"], "reasoning": "test"}


# --- Real HTTP round trip: list-returning tool, wrapped as {"items": [...]} -


@pytest.mark.integration
async def test_transport_wraps_list_returning_tool_as_items() -> None:
    async with _RunningServer(_build_list_server()) as server:
        transport = MCPHttpTransport(url=server.url, tool_name="retrieve_documents")
        result = await transport({"queries": ["q1", "q2"], "top_k": 5})

    assert result == {
        "items": [
            {"chunk_id": "q1:chunk:0000", "score": 1.0},
            {"chunk_id": "q2:chunk:0000", "score": 1.0},
        ]
    }


# --- Real HTTP round trip: remote tool failure propagates -------------------


@pytest.mark.integration
async def test_transport_propagates_remote_tool_failure() -> None:
    async with _RunningServer(_build_failing_server()) as server:
        transport = MCPHttpTransport(url=server.url, tool_name="generate_answer")
        with pytest.raises(Exception):  # fastmcp raises a ToolError for the remote exception
            await transport({"query": "q", "evidence": []})


# --- Constructor validation (no server needed) ------------------------------


@pytest.mark.unit
def test_transport_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        MCPHttpTransport(url="", tool_name="search_papers")


@pytest.mark.unit
def test_transport_rejects_empty_tool_name() -> None:
    with pytest.raises(ValueError):
        MCPHttpTransport(url="http://localhost:8765/mcp", tool_name="")
