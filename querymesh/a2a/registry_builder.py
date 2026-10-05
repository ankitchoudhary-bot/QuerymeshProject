"""Builds the standard QueryMesh agent registry.

Registers each of the five specialized agents (Phase 11) under the
capability names the spec uses: research, retrieval, grading, synthesis,
citation. `querymesh.graph` (or future MCP/A2A callers) can then discover
agents by capability instead of importing concrete classes directly.
"""

from __future__ import annotations

from querymesh.a2a.discovery import AgentCard, AgentRegistry
from querymesh.agents.citation import CitationAgent
from querymesh.agents.grader import GraderAgent
from querymesh.agents.researcher import ResearcherAgent
from querymesh.agents.retriever import RetrieverAgent
from querymesh.agents.synthesizer import SynthesizerAgent


def build_default_registry(
    researcher: ResearcherAgent,
    retriever: RetrieverAgent,
    grader: GraderAgent,
    synthesizer: SynthesizerAgent,
    citation: CitationAgent,
) -> AgentRegistry:
    registry = AgentRegistry()

    registry.register(
        AgentCard(name="researcher", capabilities=["research"], description="Plans search queries"),
        researcher,
    )
    registry.register(
        AgentCard(name="retriever", capabilities=["retrieval"], description="Executes retrieval"),
        retriever,
    )
    registry.register(
        AgentCard(name="grader", capabilities=["grading"], description="Grades document relevance"),
        grader,
    )
    registry.register(
        AgentCard(
            name="synthesizer", capabilities=["synthesis"], description="Synthesizes grounded answers"
        ),
        synthesizer,
    )
    registry.register(
        AgentCard(
            name="citation", capabilities=["citation"], description="Verifies and builds citations"
        ),
        citation,
    )

    return registry
