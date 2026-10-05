"""Document Grader agent.

Responsibilities: evaluate relevance of each retrieved document to the
user's question, filter out poor documents, score evidence quality.
Uses structured output (JSON) rather than parsing free-form strings.
"""

from __future__ import annotations

from querymesh.agents.models import DocumentGradeResult, GradedDocument
from querymesh.agents.structured_output import (
    StructuredOutputError,
    parse_json_response,
)
from querymesh.errors import AgentError
from querymesh.llm.base import LLM
from querymesh.observability.logging import get_logger
from querymesh.retrieval.models import RetrievedDocument

logger = get_logger(__name__)

_PROMPT_TEMPLATE = """You are grading whether a retrieved passage is relevant \
evidence for answering a research question.

Question: {query}

Passage:
{passage}

Respond with a JSON object only:
  - "relevant": true or false
  - "score": a relevance score from 0.0 to 1.0
  - "reason": one short sentence explaining the grade
"""

_DEFAULT_RELEVANCE_THRESHOLD = 0.5


class GraderAgent:
    """Grades retrieved documents for relevance using the LLM."""

    def __init__(self, llm: LLM, relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD) -> None:
        if llm is None:
            raise AgentError("GraderAgent requires a non-None llm")
        self.llm = llm
        self.relevance_threshold = relevance_threshold

    async def grade_one(self, query: str, document: RetrievedDocument) -> GradedDocument:
        passage_text = document.document.get("text", "")
        prompt = _PROMPT_TEMPLATE.format(query=query, passage=passage_text)
        response = await self.llm.ainvoke(prompt)

        try:
            parsed = parse_json_response(response.text)
            if not isinstance(parsed, dict):
                raise StructuredOutputError("Expected a JSON object")
            grade = DocumentGradeResult(
                relevant=bool(parsed["relevant"]),
                score=float(parsed["score"]),
                reason=str(parsed.get("reason", "")),
            )
        except (StructuredOutputError, KeyError, ValueError, TypeError) as exc:
            logger.warning(
                "Grader agent could not parse grade for %s (%s); grading as not relevant.",
                document.chunk_id,
                exc,
            )
            grade = DocumentGradeResult(relevant=False, score=0.0, reason="Could not parse grade")

        return GradedDocument(chunk_id=document.chunk_id, document=document.document, grade=grade)

    async def grade_and_filter(
        self, query: str, documents: list[RetrievedDocument]
    ) -> list[GradedDocument]:
        """Grade all documents, returning only those graded relevant AND
        scoring at or above `relevance_threshold`.
        """
        graded = [await self.grade_one(query, doc) for doc in documents]
        return [
            g for g in graded if g.grade.relevant and g.grade.score >= self.relevance_threshold
        ]
