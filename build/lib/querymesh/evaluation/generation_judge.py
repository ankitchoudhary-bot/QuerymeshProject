"""LLM-as-judge generation metrics: faithfulness and answer relevance.

These require actual judgment about meaning, not something mechanically
checkable — hence an LLM judge. Rule: "isolate the evaluator from the
production generation model where possible." `GenerationJudge` takes its
own `judge_llm`, which callers should construct as a *separate* instance
(or even a different provider/model) from whatever `SynthesizerAgent` used
to generate the answer being judged — nothing in this module reuses or
assumes the production LLM.
"""

from __future__ import annotations

from pydantic import BaseModel

from querymesh.agents.structured_output import StructuredOutputError, parse_json_response
from querymesh.llm.base import LLM
from querymesh.observability.logging import get_logger
from querymesh.retrieval.models import RetrievedDocument

logger = get_logger(__name__)

_FAITHFULNESS_PROMPT = """You are judging whether an AI-generated answer is faithful to its evidence.

An answer is faithful if every factual claim it makes is supported by the \
evidence below. An answer is unfaithful if it states something the \
evidence doesn't support, or contradicts the evidence.

Question: {query}

Evidence:
{evidence}

Answer to judge:
{answer}

Respond with JSON only:
  - "score": a faithfulness score from 0.0 (fully unsupported) to 1.0 (fully supported)
  - "reason": one short sentence explaining the score
"""

_RELEVANCE_PROMPT = """You are judging whether an AI-generated answer actually \
addresses the question asked.

Question: {query}

Answer to judge:
{answer}

Respond with JSON only:
  - "score": a relevance score from 0.0 (does not address the question) to 1.0 (fully addresses it)
  - "reason": one short sentence explaining the score
"""


class JudgeScore(BaseModel):
    score: float
    reason: str


def _format_evidence(evidence: list[RetrievedDocument]) -> str:
    if not evidence:
        return "(no evidence)"
    return "\n\n".join(f"[{e.chunk_id}] {e.document.get('text', '')}" for e in evidence)


class GenerationJudge:
    """LLM-as-judge for faithfulness and answer relevance.

    Construct with a `judge_llm` that is a separate instance from the
    production `SynthesizerAgent`'s LLM — see module docstring.
    """

    def __init__(self, judge_llm: LLM) -> None:
        self.judge_llm = judge_llm

    async def _judge(self, prompt: str, fallback_reason: str) -> JudgeScore:
        response = await self.judge_llm.ainvoke(prompt)
        try:
            parsed = parse_json_response(response.text)
            if not isinstance(parsed, dict):
                raise StructuredOutputError("Expected a JSON object")
            return JudgeScore(score=float(parsed["score"]), reason=str(parsed.get("reason", "")))
        except (StructuredOutputError, KeyError, ValueError, TypeError) as exc:
            logger.warning("Judge could not parse score (%s); defaulting to 0.0", exc)
            return JudgeScore(score=0.0, reason=fallback_reason)

    async def judge_faithfulness(
        self, query: str, answer: str, evidence: list[RetrievedDocument]
    ) -> JudgeScore:
        prompt = _FAITHFULNESS_PROMPT.format(query=query, evidence=_format_evidence(evidence), answer=answer)
        return await self._judge(prompt, fallback_reason="Could not parse faithfulness judgment")

    async def judge_answer_relevance(self, query: str, answer: str) -> JudgeScore:
        prompt = _RELEVANCE_PROMPT.format(query=query, answer=answer)
        return await self._judge(prompt, fallback_reason="Could not parse relevance judgment")
