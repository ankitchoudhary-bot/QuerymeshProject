"""Deterministic generation-quality metrics.

Rule-based, no LLM call, no network — safe to run as a normal unit test.
These measure what can be checked mechanically from the citation/evidence
data structures alone; `faithfulness` and `answer_relevance` (which
genuinely require judgment about meaning) live in `generation_judge.py`
as LLM-as-judge checks instead, kept separate per the spec's instruction
to isolate the evaluator from the production generation model.
"""

from __future__ import annotations

from pydantic import BaseModel

from querymesh.agents.models import Citation
from querymesh.retrieval.models import RetrievedDocument


class DeterministicGenerationMetrics(BaseModel):
    citation_correctness: float
    citation_completeness: float
    retrieval_usefulness: float


def citation_correctness(citations: list[Citation], evidence: list[RetrievedDocument]) -> float:
    """Fraction of citations whose chunk_id is actually backed by the
    given evidence. 1.0 means no invented citations; this is exactly the
    property `CitationAgent.verify_citations` enforces at generation
    time — this metric re-checks it independently, at evaluation time,
    against whatever citations a run actually produced.
    """
    if not citations:
        return 1.0  # nothing to be wrong about

    valid_ids = {e.chunk_id for e in evidence if e.chunk_id}
    correct = sum(1 for c in citations if c.chunk_id in valid_ids)
    return correct / len(citations)


def citation_completeness(citations: list[Citation], evidence: list[RetrievedDocument]) -> float:
    """Whether the answer cited *something*, when there was evidence to
    cite. Deliberately simple (rule 4: don't over-engineer) — a real
    claim-by-claim completeness check needs NLP the project doesn't have
    yet; this is the honest, checkable proxy: did evidence go unused
    entirely?
    """
    if not evidence:
        return 1.0  # nothing available to cite; not a completeness failure
    return 1.0 if citations else 0.0


def retrieval_usefulness(evidence: list[RetrievedDocument], citations: list[Citation]) -> float:
    """Fraction of retrieved evidence chunks that were actually cited —
    a proxy for whether retrieval surfaced useful (used) material versus
    noise the generation step ignored.
    """
    if not evidence:
        return 0.0

    cited_ids = {c.chunk_id for c in citations}
    evidence_ids = {e.chunk_id for e in evidence if e.chunk_id}
    if not evidence_ids:
        return 0.0

    used = len(cited_ids & evidence_ids)
    return used / len(evidence_ids)


def compute_deterministic_metrics(
    citations: list[Citation], evidence: list[RetrievedDocument]
) -> DeterministicGenerationMetrics:
    return DeterministicGenerationMetrics(
        citation_correctness=citation_correctness(citations, evidence),
        citation_completeness=citation_completeness(citations, evidence),
        retrieval_usefulness=retrieval_usefulness(evidence, citations),
    )
