"""FastMCP adapter for the Retriever agent."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider

from querymesh.agents.retriever import RetrieverAgent
from querymesh.errors import MCPError
from querymesh.observability.logging import get_logger
from querymesh.security.limits import validate_top_k

logger = get_logger(__name__)


def build_retrieval_server(retriever: RetrieverAgent, auth: AuthProvider | None = None) -> FastMCP:
    mcp: FastMCP = FastMCP(name="retrieval-agent", auth=auth)

    @mcp.tool
    async def retrieve_documents(queries: list[str], top_k: int = 10) -> list[dict]:
        """Retrieve and deduplicate candidate chunks for one or more queries.

        Returns a list of RetrievedDocument dicts.
        """
        try:
            top_k = validate_top_k(top_k)
            results = await retriever.retrieve(queries, top_k=top_k)
            return [r.model_dump() for r in results]
        except Exception as exc:  # noqa: BLE001 - surface as a clear MCP-layer error
            logger.error("retrieve_documents tool call failed: %s", exc)
            raise MCPError(f"retrieve_documents failed: {exc}") from exc

    return mcp
