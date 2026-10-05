# """Citation / Verification agent.

# Responsibilities: verify claims against retrieved evidence, ensure
# citations exist, and never allow an invented citation ID to slip through
# — every `Citation.chunk_id` this agent produces is checked against the
# evidence it was given.
# """

# from __future__ import annotations

# from querymesh.agents.models import Citation
# from querymesh.observability.logging import get_logger
# from querymesh.retrieval.models import RetrievedDocument

# logger = get_logger(__name__)


# class CitationAgent:
#     """Builds citations strictly from the evidence actually retrieved.

#     This first implementation is deliberately simple: it produces one citation per evidence
#     chunk actually used, rather than trying to map individual sentences of
#     the answer to individual chunks via an LLM call. That finer-grained
#     mapping is a natural target for a later optimization pass.
#     """

#     def build_citations(self, answer: str, evidence: list[RetrievedDocument]) -> list[Citation]:
        
#         citations: list[Citation] = []
#         for doc in evidence:
#             chunk_id = doc.chunk_id
#             if not chunk_id:
#                 continue
#             source = doc.document
#             citations.append(
#                 Citation(
#                     arxiv_id=source.get("arxiv_id", "unknown"),
#                     title=source.get("title", "unknown title"),
#                     chunk_id=chunk_id,
#                     claim=answer,
#                 )
#             )
#         return citations

#     def verify_citations(self, citations: list[Citation], evidence: list[RetrievedDocument]) -> list[Citation]:
#         """Drop any citation whose chunk_id isn't among the given evidence
#         — the guard against inventing citation IDs.
#         """
#         valid_ids = {doc.chunk_id for doc in evidence if doc.chunk_id}
#         verified = [c for c in citations if c.chunk_id in valid_ids]

#         dropped = len(citations) - len(verified)
#         if dropped:
#             logger.warning("Dropped %d citation(s) not backed by retrieved evidence", dropped)

#         return verified








"""Citation / Verification agent.

Responsibilities: verify claims against retrieved evidence, ensure
citations exist, and never allow an invented citation ID to slip through
— every `Citation.chunk_id` this agent produces is checked against the
evidence it was given.

Maps individual sentences of the answer to the specific evidence chunk(s)
that support them (one LLM call for the whole answer, not one per
sentence — cheaper, and lets the model see all sentences/evidence
together for more consistent judgments). This replaces the earlier
one-citation-per-evidence-chunk approach, which cited every retrieved
chunk against the *entire* answer regardless of which part of the answer
actually used it.
"""

from __future__ import annotations

from querymesh.agents.models import Citation, CitationResult
from querymesh.agents.structured_output import StructuredOutputError, parse_json_response
from querymesh.errors import AgentError
from querymesh.llm.base import LLM
from querymesh.observability.logging import get_logger
from querymesh.retrieval.models import RetrievedDocument
from querymesh.text_utils import split_sentences, _split_claim_units

logger = get_logger(__name__)

# _MAPPING_PROMPT_TEMPLATE = """You are mapping sentences in a generated answer to the \
# specific evidence chunks that support each sentence's factual claims.

# {query_line}
# Evidence chunks:
# {evidence_block}

# Sentences in the answer (numbered from 0):
# {numbered_sentences}

# For each sentence, list the chunk_id(s) (exact IDs from the evidence above) \
# that support its factual claim. A sentence with no factual claim needing \
# support (e.g. "In summary," or a transition) should map to an empty list. \
# Never invent a chunk_id that isn't listed above.

# Respond with JSON only, one entry per sentence in order:
# {{"mappings": [{{"sentence_index": 0, "chunk_ids": ["..."]}}, ...]}}
# """
_MAPPING_PROMPT_TEMPLATE = """You are mapping sentences in a generated answer
to the specific evidence chunks that directly support each sentence.

{query_line}
Evidence chunks:
{evidence_block}

Sentences in the answer (numbered from 0):
{numbered_sentences}

For each sentence, list ONLY the chunk_id(s) that directly support the
factual information stated in that sentence.

Rules:
1. Use only chunk_ids provided in the evidence above.
2. Do not invent chunk_ids.
3. Do not cite a chunk merely because it is related to the topic.
4. If a sentence contains multiple factual claims, include every chunk
   that directly supports those claims.
5. If a sentence has no factual claim requiring evidence, return an empty list.
6. Use the most specific evidence available. Prefer a table chunk when the
   answer comes from a table.
7. Return every sentence index exactly once.

Respond with JSON only:
{{"mappings": [
    {{"sentence_index": 0, "chunk_ids": ["..."]}},
    {{"sentence_index": 1, "chunk_ids": []}}
]}}
"""

def _format_evidence_block(evidence: list[RetrievedDocument]) -> str:
    if not evidence:
        return "(no evidence)"
    lines = []
    for doc in evidence:
        chunk_id = doc.chunk_id or "unknown"
        text = doc.document.get("text", "")
        lines.append(f"[{chunk_id}] {text}")
    return "\n\n".join(lines)


def _format_numbered_sentences(sentences: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))


class CitationAgent:
    """Maps answer sentences to the evidence chunk(s) that support them,
    then builds and verifies citations strictly from the evidence given.
    """

    def __init__(self, llm: LLM) -> None:
        if llm is None:
            raise AgentError("CitationAgent requires a non-None llm")
        self.llm = llm
        

    async def build_citations(self, answer: str, evidence: list[RetrievedDocument], query: str | None = None) -> CitationResult:     #-> list[Citation]:
        """Map answer sentences to retrieved evidence and buildunique source citations."""
        valid_ids = {doc.chunk_id for doc in evidence if doc.chunk_id}
        if not valid_ids or not answer.strip():
            # return []
            return CitationResult(
            answer=answer,
            citations=[],
        )

        sentences = _split_claim_units(answer)
        if not sentences:
            # return []
            return CitationResult(
            answer=answer,
            citations=[],
        )

        query_line = f"Question: {query}\n" if query else ""
        prompt = _MAPPING_PROMPT_TEMPLATE.format(
            query_line=query_line,
            evidence_block=_format_evidence_block(evidence),
            numbered_sentences=_format_numbered_sentences(sentences),
        )
        
        try:
            response = await self.llm.ainvoke(prompt)
            parsed = parse_json_response(response.text)
            if not isinstance(parsed, dict):
                raise StructuredOutputError("Expected a JSON object")
            mappings = parsed["mappings"]
            if not isinstance(mappings, list):
                raise StructuredOutputError("Expected 'mappings' to be a list")
        except (StructuredOutputError, KeyError, TypeError) as exc:
            logger.warning(
                "CitationAgent could not parse sentence mapping (%s); "
                "falling back to one citation per evidence chunk.",
                exc,
            )
            return self._fallback_citations(evidence)

        citations: list[Citation] = []
        citation_by_chunk: dict[str, Citation] = {}
        # sentence index -> citation IDs
        sentence_citations: dict[int, list[int]] = {}
        
        for mapping in mappings:
            try:
                index = int(mapping["sentence_index"])
                chunk_ids = mapping.get("chunk_ids", [])
            except (KeyError, TypeError, ValueError):
                continue
            if index < 0 or index >= len(sentences):
                continue
            if not isinstance(chunk_ids, list):
                continue

            for chunk_id in chunk_ids:
                if chunk_id not in valid_ids:
                    continue
                if chunk_id  not in citation_by_chunk:
                
                    doc = next(d for d in evidence if d.chunk_id == chunk_id)
                    source = doc.document
                    citation = Citation(
                        citation_id=len(citations)+1,
                        arxiv_id=source.get("arxiv_id", "unknown"),
                        title=source.get("title", "unknown title"),
                        chunk_id=chunk_id,
                        page=source.get("page"),
                        chunk_type=source.get("chunk_type"),
                        # sentence_index=index,
                    )
                    citation_by_chunk[chunk_id] = citation
                    citations.append(citation)
                
                citation_id = citation_by_chunk[chunk_id].citation_id

                # -------------------------------------------------
                # Associate this citation with this sentence.
                # -------------------------------------------------

                sentence_citations.setdefault(index, [])

                if citation_id not in sentence_citations[index]:
                    sentence_citations[index].append(citation_id)

        # return citations
        # ---------------------------------------------------------
        # Add [1], [2], etc. to the original answer.
        #
        # Work backwards so string positions don't interfere
        # with each other.
        # ---------------------------------------------------------

        answer_with_markers = answer

        for index in sorted(sentence_citations.keys(), reverse=True):

            sentence = sentences[index]

            citation_ids = sorted(set(sentence_citations[index]))

            if not citation_ids:
                continue

            markers = " ".join(f"[{citation_id}]" for citation_id in citation_ids)

            replacement = f"{sentence} {markers}"

            position = answer_with_markers.rfind(sentence)

            if position == -1:
                continue

            answer_with_markers = (answer_with_markers[:position] + replacement + answer_with_markers[position + len(sentence):])

        return CitationResult(answer=answer_with_markers, citations=citations, )
        

    def _fallback_citations(self, evidence: list[RetrievedDocument]) -> list[Citation]:

        citations: list[Citation] = []
        for doc in evidence:
            chunk_id = doc.chunk_id
            if not chunk_id:
                continue
            source = doc.document
            citations.append(
                Citation(
                    citation_id=len(citations) + 1,
                    arxiv_id=source.get("arxiv_id", "unknown"),
                    title=source.get("title", "unknown title"),
                    chunk_id=chunk_id,
                    page=source.get("page"),
                    chunk_type=source.get("chunk_type"),
                    # sentence_index=None,
                )
            )
        return citations

    def verify_citations(self, citations: list[Citation], evidence: list[RetrievedDocument]) -> list[Citation]:
        """Drop any citation whose chunk_id isn't among the given evidence
        — the guard against inventing citation IDs. Still enforced here
        even though `build_citations` only ever selects from `valid_ids`
        itself, since this method is also a defense-in-depth boundary for
        citations built any other way (e.g. `_fallback_citations`, or a
        future caller constructing `Citation`s directly).
        """
        valid_ids = {doc.chunk_id for doc in evidence if doc.chunk_id}
        verified = [c for c in citations if c.chunk_id in valid_ids]

        dropped = len(citations) - len(verified)
        if dropped:
            logger.warning("Dropped %d citation(s) not backed by retrieved evidence", dropped)

        return verified
