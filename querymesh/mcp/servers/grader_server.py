"""FastMCP adapter for the Document Grader agent."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider

from querymesh.agents.grader import GraderAgent
from querymesh.errors import MCPError
from querymesh.observability.logging import get_logger
from querymesh.retrieval.models import RetrievedDocument

logger = get_logger(__name__)


def build_grader_server(grader: GraderAgent, auth: AuthProvider | None = None) -> FastMCP:
    mcp: FastMCP = FastMCP(name="grader-agent", auth=auth)

    @mcp.tool
    async def grade_documents(query: str, documents: list[dict]) -> list[dict]:
        """Grade documents for relevance and return only those that pass.

        `documents` is a list of RetrievedDocument dicts (as returned by
        `retrieve_documents`). Returns a list of GradedDocument dicts.
        """
        try:
            retrieved = [RetrievedDocument(**doc) for doc in documents]
            graded = await grader.grade_and_filter(query, retrieved)
            return [g.model_dump() for g in graded]
        except Exception as exc:  # noqa: BLE001 - surface as a clear MCP-layer error
            logger.error("grade_documents tool call failed: %s", exc)
            raise MCPError(f"grade_documents failed: {exc}") from exc

    return mcp
