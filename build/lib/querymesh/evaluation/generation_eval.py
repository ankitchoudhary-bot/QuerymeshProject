"""Combined generation evaluation: deterministic metrics + LLM-judge
scores, run over a set of deterministic test cases.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from querymesh.agents.models import Citation
from querymesh.evaluation.generation_judge import GenerationJudge
from querymesh.evaluation.generation_metrics import compute_deterministic_metrics
from querymesh.retrieval.models import RetrievedDocument


class GenerationEvalCase(BaseModel):
    """One deterministic test case: a query, the answer actually
    generated for it, the evidence it was generated from, and the
    citations produced. Fixed inputs — no randomness, no live retrieval —
    so re-running the same case always evaluates the same thing.
    """

    query: str
    answer: str
    evidence: list[RetrievedDocument] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class GenerationEvalResult(BaseModel):
    query: str
    citation_correctness: float
    citation_completeness: float
    retrieval_usefulness: float
    faithfulness: float | None = None
    answer_relevance: float | None = None
    faithfulness_reason: str | None = None
    answer_relevance_reason: str | None = None


async def evaluate_generation_case(
    case: GenerationEvalCase, judge: GenerationJudge | None = None
) -> GenerationEvalResult:
    """Evaluate one case. Deterministic metrics always run; LLM-judge
    scores (faithfulness, answer_relevance) run only if a `judge` is
    given, so this works with zero LLM/network dependency when you only
    need the deterministic half.
    """
    deterministic = compute_deterministic_metrics(case.citations, case.evidence)

    result = GenerationEvalResult(
        query=case.query,
        citation_correctness=deterministic.citation_correctness,
        citation_completeness=deterministic.citation_completeness,
        retrieval_usefulness=deterministic.retrieval_usefulness,
    )

    if judge is not None:
        faithfulness = await judge.judge_faithfulness(case.query, case.answer, case.evidence)
        relevance = await judge.judge_answer_relevance(case.query, case.answer)
        result.faithfulness = faithfulness.score
        result.faithfulness_reason = faithfulness.reason
        result.answer_relevance = relevance.score
        result.answer_relevance_reason = relevance.reason

    return result


async def evaluate_generation_dataset(
    dataset: list[GenerationEvalCase], judge: GenerationJudge | None = None
) -> list[GenerationEvalResult]:
    return [await evaluate_generation_case(case, judge) for case in dataset]
