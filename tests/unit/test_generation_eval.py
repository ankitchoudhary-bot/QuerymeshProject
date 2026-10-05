"""Phase 26 tests: deterministic generation metrics (citation correctness/
completeness, retrieval usefulness), GenerationJudge (LLM-as-judge, via
FakeLLM for determinism), and the combined evaluation harness — all
deterministic test cases, no live LLM or network.
"""

from __future__ import annotations

import json

import pytest

from querymesh.agents.models import Citation
from querymesh.evaluation import (
    GenerationEvalCase,
    GenerationJudge,
    citation_completeness,
    citation_correctness,
    evaluate_generation_case,
    evaluate_generation_dataset,
    retrieval_usefulness,
)
from querymesh.llm.fake import FakeLLM
from querymesh.retrieval.models import RetrievedDocument


def _evidence(chunk_id: str, arxiv_id: str = "p1", text: str = "some evidence text") -> RetrievedDocument:
    return RetrievedDocument(
        document={"chunk_id": chunk_id, "arxiv_id": arxiv_id, "title": "T", "text": text},
        score=1.0,
        rank=1,
        retrieval_method="hybrid_rrf",
    )


def _citation(chunk_id: str, arxiv_id: str = "p1") -> Citation:
    return Citation(arxiv_id=arxiv_id, title="T", chunk_id=chunk_id, claim="the answer")


# --- citation_correctness --------------------------------------------------


@pytest.mark.unit
def test_citation_correctness_all_backed_is_one() -> None:
    evidence = [_evidence("a:chunk:0000"), _evidence("a:chunk:0001")]
    citations = [_citation("a:chunk:0000"), _citation("a:chunk:0001")]
    assert citation_correctness(citations, evidence) == 1.0


@pytest.mark.unit
def test_citation_correctness_partial_invented() -> None:
    evidence = [_evidence("a:chunk:0000")]
    citations = [_citation("a:chunk:0000"), _citation("invented:chunk:9999")]
    assert citation_correctness(citations, evidence) == pytest.approx(0.5)


@pytest.mark.unit
def test_citation_correctness_no_citations_is_one() -> None:
    assert citation_correctness([], [_evidence("a:chunk:0000")]) == 1.0


@pytest.mark.unit
def test_citation_correctness_all_invented_is_zero() -> None:
    evidence = [_evidence("a:chunk:0000")]
    citations = [_citation("invented:chunk:9999")]
    assert citation_correctness(citations, evidence) == 0.0


# --- citation_completeness -------------------------------------------------


@pytest.mark.unit
def test_citation_completeness_evidence_with_citations_is_one() -> None:
    evidence = [_evidence("a:chunk:0000")]
    citations = [_citation("a:chunk:0000")]
    assert citation_completeness(citations, evidence) == 1.0


@pytest.mark.unit
def test_citation_completeness_evidence_but_no_citations_is_zero() -> None:
    evidence = [_evidence("a:chunk:0000")]
    assert citation_completeness([], evidence) == 0.0


@pytest.mark.unit
def test_citation_completeness_no_evidence_is_one() -> None:
    assert citation_completeness([], []) == 1.0


# --- retrieval_usefulness -------------------------------------------------


@pytest.mark.unit
def test_retrieval_usefulness_all_evidence_cited() -> None:
    evidence = [_evidence("a:chunk:0000"), _evidence("a:chunk:0001")]
    citations = [_citation("a:chunk:0000"), _citation("a:chunk:0001")]
    assert retrieval_usefulness(evidence, citations) == 1.0


@pytest.mark.unit
def test_retrieval_usefulness_half_evidence_cited() -> None:
    evidence = [_evidence("a:chunk:0000"), _evidence("a:chunk:0001")]
    citations = [_citation("a:chunk:0000")]
    assert retrieval_usefulness(evidence, citations) == pytest.approx(0.5)


@pytest.mark.unit
def test_retrieval_usefulness_no_evidence_is_zero() -> None:
    assert retrieval_usefulness([], []) == 0.0


@pytest.mark.unit
def test_retrieval_usefulness_evidence_but_no_citations_is_zero() -> None:
    evidence = [_evidence("a:chunk:0000")]
    assert retrieval_usefulness(evidence, []) == 0.0


# --- GenerationJudge (FakeLLM, deterministic) ------------------------------


@pytest.mark.unit
async def test_judge_faithfulness_parses_score() -> None:
    judge_llm = FakeLLM(response_text=json.dumps({"score": 0.9, "reason": "well supported"}))
    judge = GenerationJudge(judge_llm)

    result = await judge.judge_faithfulness(
        "What is attention?", "Attention weighs input relevance.", [_evidence("a:chunk:0000")]
    )

    assert result.score == 0.9
    assert result.reason == "well supported"


@pytest.mark.unit
async def test_judge_answer_relevance_parses_score() -> None:
    judge_llm = FakeLLM(response_text=json.dumps({"score": 1.0, "reason": "directly answers"}))
    judge = GenerationJudge(judge_llm)

    result = await judge.judge_answer_relevance("What is attention?", "Attention is a mechanism...")

    assert result.score == 1.0


@pytest.mark.unit
async def test_judge_bad_json_defaults_to_zero() -> None:
    judge_llm = FakeLLM(response_text="not valid json")
    judge = GenerationJudge(judge_llm)

    result = await judge.judge_faithfulness("q", "a", [])

    assert result.score == 0.0
    assert "Could not parse" in result.reason


@pytest.mark.unit
def test_judge_uses_its_own_llm_instance_not_a_shared_one() -> None:
    """The judge's LLM is whatever the caller constructs it with — this
    test documents the isolation contract: nothing in GenerationJudge
    reaches into a production SynthesizerAgent's LLM.
    """
    production_llm = FakeLLM(response_text="production answer")
    judge_llm = FakeLLM(response_text=json.dumps({"score": 0.5, "reason": "r"}))

    judge = GenerationJudge(judge_llm)

    assert judge.judge_llm is judge_llm
    assert judge.judge_llm is not production_llm


# --- Combined harness -----------------------------------------------------


@pytest.mark.unit
async def test_evaluate_generation_case_deterministic_only() -> None:
    case = GenerationEvalCase(
        query="What is attention?",
        answer="Attention weighs input relevance.",
        evidence=[_evidence("a:chunk:0000")],
        citations=[_citation("a:chunk:0000")],
    )

    result = await evaluate_generation_case(case)  # no judge given

    assert result.citation_correctness == 1.0
    assert result.citation_completeness == 1.0
    assert result.retrieval_usefulness == 1.0
    assert result.faithfulness is None
    assert result.answer_relevance is None


@pytest.mark.unit
async def test_evaluate_generation_case_with_judge() -> None:
    case = GenerationEvalCase(
        query="What is attention?",
        answer="Attention weighs input relevance.",
        evidence=[_evidence("a:chunk:0000")],
        citations=[_citation("a:chunk:0000")],
    )
    judge_llm = FakeLLM(response_text=json.dumps({"score": 0.8, "reason": "mostly faithful"}))
    judge = GenerationJudge(judge_llm)

    result = await evaluate_generation_case(case, judge=judge)

    assert result.faithfulness == 0.8
    assert result.answer_relevance == 0.8  # same FakeLLM response for both calls
    assert result.faithfulness_reason == "mostly faithful"


@pytest.mark.unit
async def test_evaluate_generation_dataset_runs_all_cases() -> None:
    dataset = [
        GenerationEvalCase(
            query="q1",
            answer="a1",
            evidence=[_evidence("a:chunk:0000")],
            citations=[_citation("a:chunk:0000")],
        ),
        GenerationEvalCase(
            query="q2",
            answer="a2",
            evidence=[],
            citations=[],
        ),
    ]

    results = await evaluate_generation_dataset(dataset)

    assert len(results) == 2
    assert results[0].query == "q1"
    assert results[1].query == "q2"
    assert results[1].citation_completeness == 1.0  # no evidence -> trivially complete


@pytest.mark.unit
async def test_deterministic_test_case_is_reproducible() -> None:
    """Running the same fixed case twice yields identical results — the
    'deterministic test cases' requirement, made concrete.
    """
    case = GenerationEvalCase(
        query="q",
        answer="a",
        evidence=[_evidence("a:chunk:0000")],
        citations=[_citation("a:chunk:0000"), _citation("invented:chunk:9999")],
    )

    result1 = await evaluate_generation_case(case)
    result2 = await evaluate_generation_case(case)

    assert result1 == result2
    assert result1.citation_correctness == pytest.approx(0.5)
