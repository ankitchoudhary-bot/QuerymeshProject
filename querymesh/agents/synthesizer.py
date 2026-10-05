"""Synthesis agent.

Responsibilities: synthesize a grounded answer to the user's question from
retrieved evidence, without inventing facts or citing papers that weren't
retrieved.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from querymesh.agents.models import SynthesisResult
from querymesh.errors import AgentError
from querymesh.llm.base import LLM, LLMChunk
from querymesh.retrieval.models import RetrievedDocument

# _PROMPT_TEMPLATE = """You are a research assistant answering a question using \
# only the evidence below.

# Rules:
#   - Do not invent facts.
#   - Do not cite papers that are not in the evidence.
#   - Distinguish evidence from inference.
#   - Say explicitly if the evidence is insufficient to answer.
#   - Prefer primary research papers.

# Question: {query}

# Evidence:
# {evidence}

# Write a grounded answer.
# """

_PROMPT_TEMPLATE = """You are a research assistant answering a question using
only the evidence below.

Rules:
  - Do not invent facts.
  - Do not cite papers that are not in the evidence.
  - Distinguish evidence from inference.
  - Say explicitly if the evidence is insufficient to answer.
  - Prefer primary research papers.
  - Do NOT generate citation numbers such as [1], [4], [24], etc.
  - Do NOT copy citation/reference numbers from the source papers.
  - Do NOT copy bibliography references from the evidence.
  - QueryMesh will add its own citation markers after you generate the answer.
  - Return only the grounded answer, without a references or citations section.

Question: {query}

Evidence:
{evidence}

Write a grounded answer without citation markers.
"""


def _format_evidence(evidence: list[RetrievedDocument]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for doc in evidence:
        source = doc.document
        chunk_id = doc.chunk_id or "unknown"
        title = source.get("title", "unknown title")
        text = source.get("text", "")
        # lines.append(f"[{chunk_id}] ({title}): {text}")
        lines.append(
            f"EVIDENCE CHUNK ID: {chunk_id}\n"
            f"SOURCE TITLE: {title}\n"
            f"SOURCE TEXT:\n{text}"
        )

    return "\n\n".join(lines)


class SynthesizerAgent:
    """Generates a grounded answer from graded evidence."""

    def __init__(self, llm: LLM) -> None:
        if llm is None:
            raise AgentError("SynthesizerAgent requires a non-None llm")
        self.llm = llm

    async def synthesize(self, query: str, evidence: list[RetrievedDocument]) -> SynthesisResult:
        prompt = _PROMPT_TEMPLATE.format(query=query, evidence=_format_evidence(evidence))
        response = await self.llm.ainvoke(prompt)
        return SynthesisResult(answer=response.text, evidence_sufficient=bool(evidence))

    async def synthesize_streaming(self, query: str, evidence: list[RetrievedDocument]) -> AsyncIterator[LLMChunk | SynthesisResult]:
        """Stream the answer token by token, then yield the final
        `SynthesisResult` as the last item.

        Callers distinguish the two by type: every `LLMChunk` is a token;
        the single trailing `SynthesisResult` is the accumulated answer.
        """
        prompt = _PROMPT_TEMPLATE.format(query=query, evidence=_format_evidence(evidence))
        pieces: list[str] = []

        async for chunk in self.llm.astream(prompt):
            pieces.append(chunk.text)
            yield chunk

        yield SynthesisResult(answer="".join(pieces), evidence_sufficient=bool(evidence))
