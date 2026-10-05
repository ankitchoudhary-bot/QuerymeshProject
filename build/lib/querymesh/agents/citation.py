"""Citation / Verification agent.

Responsibilities: verify claims against retrieved evidence, ensure
citations exist, and never allow an invented citation ID to slip through
— every `Citation.chunk_id` this agent produces is checked against the
evidence it was given.
"""

from __future__ import annotations

from querymesh.agents.models import Citation
from querymesh.observability.logging import get_logger
from querymesh.retrieval.models import RetrievedDocument

logger = get_logger(__name__)


class CitationAgent:
    """Builds citations strictly from the evidence actually retrieved.

    This first implementation is deliberately simple (rule 4: make it
    correct, don't over-engineer): it produces one citation per evidence
    chunk actually used, rather than trying to map individual sentences of
    the answer to individual chunks via an LLM call. That finer-grained
    mapping is a natural target for a later optimization pass.
    """

    def build_citations(
        self, answer: str, evidence: list[RetrievedDocument]
    ) -> list[Citation]:
        citations: list[Citation] = []
        for doc in evidence:
            chunk_id = doc.chunk_id
            if not chunk_id:
                continue
            source = doc.document
            citations.append(
                Citation(
                    arxiv_id=source.get("arxiv_id", "unknown"),
                    title=source.get("title", "unknown title"),
                    chunk_id=chunk_id,
                    claim=answer,
                )
            )
        return citations

    def verify_citations(
        self, citations: list[Citation], evidence: list[RetrievedDocument]
    ) -> list[Citation]:
        """Drop any citation whose chunk_id isn't among the given evidence
        — the guard against inventing citation IDs.
        """
        valid_ids = {doc.chunk_id for doc in evidence if doc.chunk_id}
        verified = [c for c in citations if c.chunk_id in valid_ids]

        dropped = len(citations) - len(verified)
        if dropped:
            logger.warning("Dropped %d citation(s) not backed by retrieved evidence", dropped)

        return verified
