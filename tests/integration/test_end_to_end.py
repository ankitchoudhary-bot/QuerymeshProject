"""The end-to-end test: user query -> graph -> retrieval -> grading ->
generation -> citation -> final answer, all through the real compiled
LangGraph workflow (Phase 13/14), with real component composition
(RetrievalService -> HybridRetriever -> BM25 + Vector -> RRF, Phase 5-10)
and only the outermost boundaries faked: the OpenSearch transport and the
LLM provider.
"""

from __future__ import annotations

import json

import pytest

from querymesh.embeddings.fake import FakeEmbedder
from querymesh.graph import compile_workflow, create_initial_state
from querymesh.indexing.indexer import ChunkIndexer
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.llm.base import LLMResponse
from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever
from tests.integration.test_retrieval_integration import _FakeOpenSearchClient


class _ScriptedLLM:
    """Returns each response in order; repeats the last once exhausted."""

    model = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._index = 0

    async def ainvoke(self, prompt: str) -> LLMResponse:
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return LLMResponse(text=text, model=self.model)

    async def astream(self, prompt: str):
        from querymesh.llm.base import LLMChunk

        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        yield LLMChunk(text=text, model=self.model, is_final=True)


def _chunk(index: int, text: str, arxiv_id: str = "2608.13558v1") -> PaperChunk:
    return PaperChunk(
        chunk_id=f"{arxiv_id}:chunk:{index:04d}",
        arxiv_id=arxiv_id,
        title="Attention Evolutions",
        section="Body",
        text=text,
        page=None,
        chunk_index=index,
    )


@pytest.mark.integration
async def test_end_to_end_query_to_grounded_answer() -> None:
    """The full pipeline, real components throughout:

    user query -> ResearcherAgent (plans queries)
               -> RetrieverAgent -> RetrievalService -> HybridRetriever
                  -> BM25Retriever + VectorRetriever -> RRF
                  (reading real indexed documents from a fake OpenSearch store)
               -> GraderAgent (filters for relevance)
               -> SynthesizerAgent (grounded answer)
               -> CitationAgent (verified citations)
               -> final answer
    """
    # --- Arrange: a real, populated retrieval stack -----------------------
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-e2e")
    await conn.create_index()

    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)
    await indexer.index_chunks(
        [
            _chunk(0, "Attention mechanisms allow models to weigh input relevance."),
            _chunk(1, "The original Transformer used scaled dot-product attention."),
            _chunk(2, "Efficient attention variants reduce quadratic complexity."),
        ]
    )

    retrieval_service = RetrievalService(
        hybrid_retriever=HybridRetriever(
            bm25_retriever=BM25Retriever(conn),
            vector_retriever=VectorRetriever(conn),
            embedder=embedder,
        )
    )

    # --- Arrange: a scripted LLM driving research -> grade x3 -> synthesize
    llm = _ScriptedLLM(
        [
            json.dumps({"search_queries": ["attention mechanisms"], "reasoning": "plan"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "on topic"}),  # grade doc 1
            json.dumps({"relevant": True, "score": 0.85, "reason": "on topic"}),  # grade doc 2
            json.dumps({"relevant": False, "score": 0.1, "reason": "off topic"}),  # grade doc 3
            "Attention mechanisms evolved from full dot-product attention in the "
            "original Transformer toward more efficient variants that reduce "
            "quadratic complexity.",
        ]
    )

    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)  # type: ignore[arg-type]

    # --- Act ----------------------------------------------------------------
    state = create_initial_state(
        "How have attention mechanisms evolved since the original Transformer?",
        max_retries=1,
    )
    result = await compiled.ainvoke(state)

    # --- Assert: the full chain produced a grounded, cited answer -----------
    assert result["search_queries"] == ["attention mechanisms"]
    assert len(result["retrieved_documents"]) >= 1
    assert len(result["evidence"]) == 2  # exactly the 2 docs graded relevant
    assert "efficient variants" in result["answer"]
    assert len(result["citations"]) == 2

    cited_chunk_ids = {c.chunk_id for c in result["citations"]}
    evidence_chunk_ids = {e.chunk_id for e in result["evidence"]}
    assert cited_chunk_ids == evidence_chunk_ids  # every citation is backed by real evidence
    assert result["errors"] == []
    assert result["retry_count"] == 0  # evidence was sufficient first pass


@pytest.mark.integration
async def test_end_to_end_gracefully_handles_no_relevant_evidence() -> None:
    """Full pipeline where every document is graded irrelevant: the graph
    should still terminate cleanly (bounded retries) with an empty-evidence
    answer rather than crashing.
    """
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-e2e-2")
    await conn.create_index()

    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)
    await indexer.index_chunks([_chunk(0, "Completely unrelated content about gardening.")])

    retrieval_service = RetrievalService(
        hybrid_retriever=HybridRetriever(
            bm25_retriever=BM25Retriever(conn),
            vector_retriever=VectorRetriever(conn),
            embedder=embedder,
        )
    )

    llm = _ScriptedLLM(
        [
            json.dumps({"search_queries": ["quantum computing"], "reasoning": "plan"}),
            json.dumps({"relevant": False, "score": 0.0, "reason": "off topic"}),
            "No relevant evidence was found for this question.",
        ]
    )

    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)  # type: ignore[arg-type]
    state = create_initial_state("What is quantum computing?", max_retries=1)
    result = await compiled.ainvoke(state)

    assert result["evidence"] == []
    assert result["citations"] == []
    assert result["retry_count"] == 1  # exhausted its one retry, then finished anyway
