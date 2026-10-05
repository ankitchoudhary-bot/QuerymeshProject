"""FastMCP adapter for the Citation / Verification agent."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider

from querymesh.agents.citation import CitationAgent
from querymesh.errors import MCPError
from querymesh.observability.logging import get_logger
from querymesh.retrieval.models import RetrievedDocument

logger = get_logger(__name__)


def build_citation_server(citation_agent: CitationAgent, auth: AuthProvider | None = None) -> FastMCP:
    mcp: FastMCP = FastMCP(name="citation-agent", auth=auth)

    @mcp.tool
    async def verify_citations(answer: str, evidence: list[dict]) -> list[dict]:
        """Build and verify citations for an answer against its evidence.

        Any citation not backed by the given evidence is dropped. Returns
        a list of Citation dicts.
        """
        try:
            docs = [RetrievedDocument(**doc) for doc in evidence]
            citations = citation_agent.build_citations(answer, docs)
            verified = citation_agent.verify_citations(citations, docs)
            return [c.model_dump() for c in verified]
        except Exception as exc:  # noqa: BLE001 - surface as a clear MCP-layer error
            logger.error("verify_citations tool call failed: %s", exc)
            raise MCPError(f"verify_citations failed: {exc}") from exc

    return mcp
