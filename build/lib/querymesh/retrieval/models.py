"""Shared retrieval result model.

Used by BM25 (this phase), vector search (Phase 8), and RRF (Phase 9) so
that fusion code never needs to know which retrieval method produced a
given result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """One retrieved chunk, with its score/rank from a specific method."""

    document: dict = Field(description="The raw indexed document (_source)")
    score: float
    rank: int
    retrieval_method: str  # "bm25" | "vector" | "hybrid_rrf" (later phases)

    @property
    def chunk_id(self) -> str | None:
        return self.document.get("chunk_id")
