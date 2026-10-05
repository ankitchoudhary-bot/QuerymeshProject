"""FastMCP adapter for the Synthesis agent."""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AuthProvider

from querymesh.agents.synthesizer import SynthesizerAgent
from querymesh.errors import MCPError
from querymesh.observability.logging import get_logger
from querymesh.retrieval.models import RetrievedDocument

logger = get_logger(__name__)


def build_synthesis_server(synthesizer: SynthesizerAgent, auth: AuthProvider | None = None) -> FastMCP:
    mcp: FastMCP = FastMCP(name="synthesis-agent", auth=auth)

    @mcp.tool
    async def generate_answer(query: str, evidence: list[dict]) -> dict:
        """Synthesize a grounded answer from evidence.

        `evidence` is a list of RetrievedDocument dicts. Returns
        {"answer": str, "evidence_sufficient": bool}.
        """
        try:
            docs = [RetrievedDocument(**doc) for doc in evidence]
            result = await synthesizer.synthesize(query, docs)
            return result.model_dump()
        except Exception as exc:  # noqa: BLE001 - surface as a clear MCP-layer error
            logger.error("generate_answer tool call failed: %s", exc)
            raise MCPError(f"generate_answer failed: {exc}") from exc

    return mcp
