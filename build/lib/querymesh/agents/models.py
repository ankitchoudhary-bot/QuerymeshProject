"""Structured models shared across agents.

Agents communicate through these rather than raw strings wherever an LLM
is asked for something structured — matches the project's "use structured
output rather than parsing arbitrary strings" rule (for the grader) and
extends the same discipline to the other agents' outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    """Output of the Researcher agent: what to search for."""

    search_queries: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class DocumentGradeResult(BaseModel):
    """Output of the Grader agent for a single document."""

    relevant: bool
    score: float
    reason: str


class GradedDocument(BaseModel):
    """A retrieved document paired with its grade."""

    chunk_id: str | None
    document: dict
    grade: DocumentGradeResult


class SynthesisResult(BaseModel):
    """Output of the Synthesis agent."""

    answer: str
    evidence_sufficient: bool = True


class Citation(BaseModel):
    """A single citation grounding a claim in retrieved evidence."""

    arxiv_id: str
    title: str
    chunk_id: str
    claim: str
