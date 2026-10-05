# """FastMCP adapter for the Researcher agent.

#   Agent
#    ↑
#   MCP Adapter (this module)
#    ↑
#   FastMCP Server

# The adapter only translates between MCP's JSON-friendly tool signatures
# and the agent's own Pydantic models — no business logic lives here.
# """

# from __future__ import annotations

# from fastmcp import FastMCP
# from fastmcp.server.auth.auth import AuthProvider

# from querymesh.agents.researcher import ResearcherAgent
# from querymesh.errors import MCPError
# from querymesh.observability.logging import get_logger
# from querymesh.security.limits import validate_query

# logger = get_logger(__name__)


# def build_research_server(researcher: ResearcherAgent, auth: AuthProvider | None = None) -> FastMCP:
#     mcp: FastMCP = FastMCP(name="research-agent", auth=auth)

#     @mcp.tool
#     async def search_papers(query: str) -> dict:
#         """Plan search queries for a research question.

#         Returns {"search_queries": [...], "reasoning": str | None}.
#         """
#         try:
#             query = validate_query(query)
#             plan = await researcher.plan(query)
#             return plan.model_dump()
#         except Exception as exc:  # noqa: BLE001 - surface as a clear MCP-layer error
#             logger.error("search_papers tool call failed: %s", exc)
#             raise MCPError(f"search_papers failed: {exc}") from exc

#     return mcp


"""FastMCP adapter for the Researcher agent.

  Agent
   ↑
  MCP Adapter (this module)
   ↑
  FastMCP Server

The adapter only translates between MCP's JSON-friendly tool signatures
and the agent's own Pydantic models — no business logic lives here.
"""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider

from querymesh.agents.researcher import ResearcherAgent
from querymesh.errors import MCPError
from querymesh.observability.logging import get_logger
from querymesh.security.limits import validate_query

logger = get_logger(__name__)


def build_research_server(researcher: ResearcherAgent, auth: AuthProvider | None = None) -> FastMCP:
    mcp: FastMCP = FastMCP(name="research-agent", auth=auth)

    @mcp.tool
    async def search_papers(query: str) -> dict:
        """Plan search queries for a research question.

        Returns {"search_queries": [...], "reasoning": str | None}.
        """
        try:
            query = validate_query(query)
            plan = await researcher.plan(query)
            return plan.model_dump()
        except Exception as exc:
            logger.error("search_papers tool call failed: %s", exc)
            raise MCPError(f"search_papers failed: {exc}") from exc

    @mcp.tool
    async def rewrite_search_queries(query: str, previous_queries: list[str]) -> dict:
        """Propose new search queries after a previous round found
        insufficient evidence (Phase 14's Advanced Agentic RAG retry loop).

        Returns {"search_queries": [...], "reasoning": str | None}.
        """
        try:
            query = validate_query(query)
            plan = await researcher.rewrite_queries(query, previous_queries)
            return plan.model_dump()
        except Exception as exc:
            logger.error("rewrite_search_queries tool call failed: %s", exc)
            raise MCPError(f"rewrite_search_queries failed: {exc}") from exc

    return mcp
