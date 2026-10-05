from __future__ import annotations

import asyncio
import socket

import pytest
from fastmcp import Client

from querymesh.mcp.servers.gateway_server import build_gateway_server
from querymesh.agents.researcher import ResearcherAgent
from querymesh.llm import get_llm

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def wait_for_server(
    host: str,
    port: int,
    timeout: float = 5.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout

    while asyncio.get_running_loop().time() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(0.05)

    raise TimeoutError(
        f"MCP gateway did not start on {host}:{port}"
    )


@pytest.mark.asyncio
async def test_gateway_sse_exposes_all_agents():
    gateway = build_gateway_server(
        researcher=ResearcherAgent(llm=get_llm()),
        grader=...,
        retriever=...,
        citation=...,
        synthesizer=...,
    )

    host = "127.0.0.1"
    port = get_free_port()

    server_task = asyncio.create_task(
        gateway.run_async(transport="sse", host=host, port=port, )
    )

    try:
        await wait_for_server(host, port)

        async with Client( f"http://{host}:{port}/sse") as client:

            tools = await client.list_tools()

            tool_names = {tool.name for tool in tools}

            assert {
                "research_search_papers",
                "grader_grade_documents",
                "retrieval_retrieve_documents",
                "citation_verify_citations",
                "synthesis_generate_answer",
            }.issubset(tool_names)
            
            result = await client.call_tool(
                "research_search_papers",
                {
                    "query": "How does self-attention work in transformers?"
                },
            )

            assert result is not None

            print(result)  

    finally:
        server_task.cancel()

        try:
            await server_task
        except asyncio.CancelledError:
            pass
        
        
# result = await client.call_tool(
#     "research_search_papers",
#     {
#         "query": "How does self-attention work in transformers?"
#     },
# )

# assert result is not None

# print(result)