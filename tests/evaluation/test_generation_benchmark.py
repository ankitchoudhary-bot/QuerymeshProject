# """The real generation evaluation: deterministic metrics computed against
# actual answers/citations produced by a real compiled graph run (not
# hand-crafted GenerationEvalCase fixtures) — mirrors Phase 25's approach of
# measuring the real pipeline rather than illustrative examples.
# """

# from __future__ import annotations

# import json

# import pytest

# from querymesh.embeddings.fake import FakeEmbedder
# from querymesh.evaluation import GenerationEvalCase, evaluate_generation_case
# from querymesh.graph import compile_workflow, create_initial_state
# from querymesh.indexing.indexer import ChunkIndexer
# from querymesh.indexing.opensearch import OpenSearchConnection
# from querymesh.ingestion.chunk_models import PaperChunk
# from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever
# from tests.integration.test_end_to_end import _ScriptedLLM
# from tests.integration.test_retrieval_integration import _FakeOpenSearchClient


# @pytest.mark.evaluation
# async def test_generation_metrics_on_real_graph_output() -> None:
#     client = _FakeOpenSearchClient()
#     conn = OpenSearchConnection(client=client, index_name="querymesh-gen-eval")
#     await conn.create_index()

#     embedder = FakeEmbedder(dimension=16)
#     indexer = ChunkIndexer(connection=conn, embedder=embedder)
#     await indexer.index_chunks(
#         [
#             PaperChunk(
#                 chunk_id="p1:chunk:0000",
#                 arxiv_id="p1",
#                 title="Attention Evolutions",
#                 section="Body",
#                 text="Attention mechanisms allow models to weigh input relevance.",
#                 page=None,
#                 chunk_index=0,
#             ),
#             PaperChunk(
#                 chunk_id="p1:chunk:0001",
#                 arxiv_id="p1",
#                 title="Attention Evolutions",
#                 section="Body",
#                 text="Efficient attention variants reduce quadratic complexity.",
#                 page=None,
#                 chunk_index=1,
#             ),
#         ]
#     )

#     retrieval_service = RetrievalService(
#         hybrid_retriever=HybridRetriever(
#             bm25_retriever=BM25Retriever(conn),
#             vector_retriever=VectorRetriever(conn),
#             embedder=embedder,
#         )
#     )

#     llm = _ScriptedLLM(
#         [
#             json.dumps({"search_queries": ["attention"], "reasoning": "plan"}),
#             json.dumps({"relevant": True, "score": 0.9, "reason": "on topic"}),
#             json.dumps({"relevant": True, "score": 0.85, "reason": "on topic"}),
#             "Attention mechanisms evolved to more efficient, less quadratic forms.",
#         ]
#     )

#     compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)  # type: ignore[arg-type]
#     query = "How have attention mechanisms evolved?"
#     result = await compiled.ainvoke(create_initial_state(query, max_retries=1))

#     case = GenerationEvalCase(
#         query=query,
#         answer=result["answer"],
#         evidence=result["evidence"],
#         citations=result["citations"],
#     )
#     eval_result = await evaluate_generation_case(case)  # deterministic metrics only

#     # Real, checkable properties of this real run: CitationAgent.verify_citations
#     # already guarantees correctness=1.0 at generation time (Phase 11); this
#     # confirms the guarantee held for this actual run's output.
#     assert eval_result.citation_correctness == 1.0
#     assert eval_result.citation_completeness == 1.0
#     assert eval_result.retrieval_usefulness == 1.0  # both graded-relevant chunks were cited


"""The real generation evaluation: deterministic metrics computed against
actual answers/citations produced by a real compiled graph run (not
hand-crafted GenerationEvalCase fixtures) — mirrors Phase 25's approach of
measuring the real pipeline rather than illustrative examples.
"""

from __future__ import annotations

import json

import pytest

from querymesh.embeddings.fake import FakeEmbedder
from querymesh.evaluation import GenerationEvalCase, evaluate_generation_case
from querymesh.graph import compile_workflow, create_initial_state
from querymesh.indexing.indexer import ChunkIndexer
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.retrieval import BM25Retriever, HybridRetriever, RetrievalService, VectorRetriever
from tests.integration.test_end_to_end import _ScriptedLLM
from tests.integration.test_retrieval_integration import _FakeOpenSearchClient


@pytest.mark.evaluation
async def test_generation_metrics_on_real_graph_output() -> None:
    client = _FakeOpenSearchClient()
    conn = OpenSearchConnection(client=client, index_name="querymesh-gen-eval")
    await conn.create_index()

    embedder = FakeEmbedder(dimension=16)
    indexer = ChunkIndexer(connection=conn, embedder=embedder)
    await indexer.index_chunks(
        [
            PaperChunk(
                chunk_id="p1:chunk:0000",
                arxiv_id="p1",
                title="Attention Evolutions",
                section="Body",
                text="Attention mechanisms allow models to weigh input relevance.",
                page=None,
                chunk_index=0,
            ),
            PaperChunk(
                chunk_id="p1:chunk:0001",
                arxiv_id="p1",
                title="Attention Evolutions",
                section="Body",
                text="Efficient attention variants reduce quadratic complexity.",
                page=None,
                chunk_index=1,
            ),
        ]
    )

    retrieval_service = RetrievalService(
        hybrid_retriever=HybridRetriever(
            bm25_retriever=BM25Retriever(conn),
            vector_retriever=VectorRetriever(conn),
            embedder=embedder,
        )
    )

    llm = _ScriptedLLM(
        [
            json.dumps({"search_queries": ["attention"], "reasoning": "plan"}),
            json.dumps({"relevant": True, "score": 0.9, "reason": "on topic"}),
            json.dumps({"relevant": True, "score": 0.85, "reason": "on topic"}),
            "Attention mechanisms evolved to more efficient, less quadratic forms.",
        ]
    )

    compiled = compile_workflow(llm=llm, retrieval_service=retrieval_service)  # type: ignore[arg-type]
    query = "How have attention mechanisms evolved?"
    result = await compiled.ainvoke(create_initial_state(query, max_retries=1))

    case = GenerationEvalCase(
        query=query,
        answer=result["answer"],
        evidence=result["evidence"],
        citations=result["citations"],
    )
    eval_result = await evaluate_generation_case(case)  # deterministic metrics only

    # Real, checkable properties of this real run: CitationAgent.verify_citations
    # already guarantees correctness=1.0 at generation time (Phase 11); this
    # confirms the guarantee held for this actual run's output.
    assert eval_result.citation_correctness == 1.0
    assert eval_result.citation_completeness == 1.0
    assert eval_result.retrieval_usefulness == 1.0  # both graded-relevant chunks were cited
