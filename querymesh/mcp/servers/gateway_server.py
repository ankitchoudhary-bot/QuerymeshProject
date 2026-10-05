from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider

from querymesh.agents.citation import CitationAgent
from querymesh.agents.grader import GraderAgent
from querymesh.agents.researcher import ResearcherAgent
from querymesh.agents.retriever import RetrieverAgent
from querymesh.agents.synthesizer import SynthesizerAgent

from querymesh.mcp.servers.research_server import build_research_server
from querymesh.mcp.servers.grader_server import build_grader_server
from querymesh.mcp.servers.retrieval_server import build_retrieval_server
from querymesh.mcp.servers.citation_server import build_citation_server
from querymesh.mcp.servers.synthesis_server import build_synthesis_server


def build_gateway_server(
    researcher: ResearcherAgent,
    grader: GraderAgent,
    retriever: RetrieverAgent,
    citation: CitationAgent,
    synthesizer: SynthesizerAgent,
    auth: AuthProvider | None = None,
) -> FastMCP:
    """Build a single MCP gateway exposing all QueryMesh agents."""

    gateway = FastMCP(
        name="querymesh-gateway",
        auth=auth,
    )

    # Build the existing independent MCP servers.
    research_server = build_research_server(researcher)
    grader_server = build_grader_server(grader)
    retrieval_server = build_retrieval_server(retriever)
    citation_server = build_citation_server(citation)
    synthesis_server = build_synthesis_server(synthesizer)

    # Mount each server under its own namespace.
    gateway.mount(
        research_server,
        namespace="research",
    )

    gateway.mount(
        grader_server,
        namespace="grader",
    )

    gateway.mount(
        retrieval_server,
        namespace="retrieval",
    )

    gateway.mount(
        citation_server,
        namespace="citation",
    )

    gateway.mount(
        synthesis_server,
        namespace="synthesis",
    )

    return gateway