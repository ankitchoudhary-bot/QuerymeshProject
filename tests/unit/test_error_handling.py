# """Phase 23 tests: the QueryMeshError hierarchy, retryable/non-retryable
# distinction, and the new AgentError/MCPError/RetrievalError wiring.
# """

# from __future__ import annotations

# import json

# import pytest

# from querymesh.a2a.errors import A2AAgentNotFoundError, A2ALoopDetectedError, A2ATimeoutError, A2AError
# from querymesh.agents import CitationAgent, GraderAgent, ResearcherAgent, RetrieverAgent, SynthesizerAgent
# from querymesh.cache.errors import CacheError
# from querymesh.embeddings.base import EmbeddingDimensionError, EmbeddingError
# from querymesh.errors import AgentError, ConfigurationError, MCPError, QueryMeshError, RetrievalError
# from querymesh.ingestion.errors import ArxivError, IngestionError
# from querymesh.indexing.errors import BulkIndexError, IndexingError
# from querymesh.llm.base import LLMConfigurationError, LLMError
# from querymesh.llm.fake import FakeLLM


# # --- Hierarchy: everything descends from QueryMeshError -----------------


# @pytest.mark.parametrize(
#     "error_cls",
#     [
#         ConfigurationError,
#         RetrievalError,
#         AgentError,
#         MCPError,
#         IngestionError,
#         ArxivError,
#         EmbeddingError,
#         EmbeddingDimensionError,
#         LLMError,
#         LLMConfigurationError,
#         IndexingError,
#         A2AError,
#         A2ATimeoutError,
#         A2AAgentNotFoundError,
#         A2ALoopDetectedError,
#         CacheError,
#     ],
# )
# @pytest.mark.unit
# def test_all_errors_descend_from_query_mesh_error(error_cls) -> None:
#     assert issubclass(error_cls, QueryMeshError)
#     assert issubclass(error_cls, Exception)


# @pytest.mark.unit
# def test_query_mesh_error_carries_retryable_flag() -> None:
#     err = QueryMeshError("boom", retryable=True)
#     assert err.retryable is True
#     assert str(err) == "boom"


# # --- Retryable / non-retryable is correctly distinguished per class -----


# @pytest.mark.unit
# def test_configuration_error_never_retryable() -> None:
#     assert ConfigurationError("bad config").retryable is False


# @pytest.mark.unit
# def test_llm_configuration_error_never_retryable() -> None:
#     assert LLMConfigurationError("missing key").retryable is False


# @pytest.mark.unit
# def test_llm_error_retryable_by_default() -> None:
#     assert LLMError("transient").retryable is True


# @pytest.mark.unit
# def test_embedding_dimension_error_never_retryable() -> None:
#     assert EmbeddingDimensionError("mismatch").retryable is False


# @pytest.mark.unit
# def test_embedding_error_retryable_by_default() -> None:
#     assert EmbeddingError("transient").retryable is True


# @pytest.mark.unit
# def test_a2a_timeout_retryable_but_agent_not_found_and_loop_are_not() -> None:
#     assert A2ATimeoutError("slow").retryable is True
#     assert A2AAgentNotFoundError("unknown").retryable is False
#     assert A2ALoopDetectedError("cycle").retryable is False


# @pytest.mark.unit
# def test_bulk_index_error_retryable_and_carries_failed_items() -> None:
#     err = BulkIndexError("2 failed", failed_items=[{"id": "a"}, {"id": "b"}])
#     assert err.retryable is True
#     assert err.failed_items == [{"id": "a"}, {"id": "b"}]


# @pytest.mark.unit
# def test_cache_error_retryable_by_default() -> None:
#     assert CacheError("redis blip").retryable is True


# @pytest.mark.unit
# def test_retrieval_error_retryable_by_default() -> None:
#     assert RetrievalError("transient").retryable is True


# @pytest.mark.unit
# def test_agent_error_not_retryable_by_default() -> None:
#     assert AgentError("bad input").retryable is False


# @pytest.mark.unit
# def test_mcp_error_retryable_by_default() -> None:
#     assert MCPError("transient").retryable is True


# # --- AgentError guard clauses on agent constructors -----------------------


# @pytest.mark.unit
# def test_researcher_agent_rejects_none_llm() -> None:
#     with pytest.raises(AgentError):
#         ResearcherAgent(None)  # type: ignore[arg-type]


# @pytest.mark.unit
# def test_grader_agent_rejects_none_llm() -> None:
#     with pytest.raises(AgentError):
#         GraderAgent(None)  # type: ignore[arg-type]


# @pytest.mark.unit
# def test_synthesizer_agent_rejects_none_llm() -> None:
#     with pytest.raises(AgentError):
#         SynthesizerAgent(None)  # type: ignore[arg-type]


# @pytest.mark.unit
# def test_retriever_agent_rejects_none_retrieval_service() -> None:
#     with pytest.raises(AgentError):
#         RetrieverAgent(None)  # type: ignore[arg-type]


# @pytest.mark.unit
# def test_agent_constructors_accept_valid_dependencies() -> None:
#     llm = FakeLLM()
#     ResearcherAgent(llm)
#     GraderAgent(llm)
#     SynthesizerAgent(llm)
#     RetrieverAgent(object())  # type: ignore[arg-type]
#     CitationAgent()  # no dependencies, always fine


# # --- MCPError wraps unexpected agent failures at the MCP boundary ---------


# @pytest.mark.unit
# async def test_mcp_server_wraps_unexpected_agent_failure() -> None:
#     from fastmcp import Client

#     from querymesh.mcp.servers import build_research_server

#     class _BrokenLLM:
#         model = "broken"

#         async def ainvoke(self, prompt: str):
#             raise RuntimeError("provider is down")

#     server = build_research_server(ResearcherAgent(_BrokenLLM()))  # type: ignore[arg-type]

#     async with Client(server) as client:
#         result = await client.call_tool("search_papers", {"query": "q"}, raise_on_error=False)

#     assert result.is_error is True


# @pytest.mark.unit
# async def test_mcp_server_succeeds_normally_without_wrapping_interfering() -> None:
#     from fastmcp import Client

#     from querymesh.mcp.servers import build_research_server

#     llm = FakeLLM(response_text=json.dumps({"search_queries": ["q1"]}))
#     server = build_research_server(ResearcherAgent(llm))

#     async with Client(server) as client:
#         result = await client.call_tool("search_papers", {"query": "q"})

#     assert result.data["search_queries"] == ["q1"]


# # --- RetrievalService wraps unexpected retriever failures ------------------


# @pytest.mark.unit
# async def test_retrieval_service_wraps_unexpected_failure() -> None:
#     from querymesh.retrieval.service import RetrievalService

#     class _BrokenHybridRetriever:
#         async def search(self, query, top_k=10, filters=None):
#             raise RuntimeError("cluster unreachable")

#     service = RetrievalService(hybrid_retriever=_BrokenHybridRetriever())  # type: ignore[arg-type]

#     with pytest.raises(RetrievalError) as exc_info:
#         await service.retrieve("q")

#     assert exc_info.value.retryable is True


# @pytest.mark.unit
# async def test_retrieval_service_does_not_rewrap_query_mesh_errors() -> None:
#     from querymesh.retrieval.service import RetrievalService

#     class _RetrieverRaisingTypedError:
#         async def search(self, query, top_k=10, filters=None):
#             raise IndexingError("already well-typed")

#     service = RetrievalService(hybrid_retriever=_RetrieverRaisingTypedError())  # type: ignore[arg-type]

#     with pytest.raises(IndexingError):  # not wrapped in a generic RetrievalError
#         await service.retrieve("q")


"""Phase 23 tests: the QueryMeshError hierarchy, retryable/non-retryable
distinction, and the new AgentError/MCPError/RetrievalError wiring.
"""

from __future__ import annotations

import json

import pytest

from querymesh.a2a.errors import A2AAgentNotFoundError, A2ALoopDetectedError, A2ATimeoutError, A2AError
from querymesh.agents import CitationAgent, GraderAgent, ResearcherAgent, RetrieverAgent, SynthesizerAgent
from querymesh.cache.errors import CacheError
from querymesh.embeddings.base import EmbeddingDimensionError, EmbeddingError
from querymesh.errors import AgentError, ConfigurationError, MCPError, QueryMeshError, RetrievalError
from querymesh.ingestion.errors import ArxivError, IngestionError
from querymesh.indexing.errors import BulkIndexError, IndexingError
from querymesh.llm.base import LLMConfigurationError, LLMError
from querymesh.llm.fake import FakeLLM


# --- Hierarchy: everything descends from QueryMeshError -----------------


@pytest.mark.parametrize(
    "error_cls",
    [
        ConfigurationError,
        RetrievalError,
        AgentError,
        MCPError,
        IngestionError,
        ArxivError,
        EmbeddingError,
        EmbeddingDimensionError,
        LLMError,
        LLMConfigurationError,
        IndexingError,
        A2AError,
        A2ATimeoutError,
        A2AAgentNotFoundError,
        A2ALoopDetectedError,
        CacheError,
    ],
)
@pytest.mark.unit
def test_all_errors_descend_from_query_mesh_error(error_cls) -> None:
    assert issubclass(error_cls, QueryMeshError)
    assert issubclass(error_cls, Exception)


@pytest.mark.unit
def test_query_mesh_error_carries_retryable_flag() -> None:
    err = QueryMeshError("boom", retryable=True)
    assert err.retryable is True
    assert str(err) == "boom"


# --- Retryable / non-retryable is correctly distinguished per class -----


@pytest.mark.unit
def test_configuration_error_never_retryable() -> None:
    assert ConfigurationError("bad config").retryable is False


@pytest.mark.unit
def test_llm_configuration_error_never_retryable() -> None:
    assert LLMConfigurationError("missing key").retryable is False


@pytest.mark.unit
def test_llm_error_retryable_by_default() -> None:
    assert LLMError("transient").retryable is True


@pytest.mark.unit
def test_embedding_dimension_error_never_retryable() -> None:
    assert EmbeddingDimensionError("mismatch").retryable is False


@pytest.mark.unit
def test_embedding_error_retryable_by_default() -> None:
    assert EmbeddingError("transient").retryable is True


@pytest.mark.unit
def test_a2a_timeout_retryable_but_agent_not_found_and_loop_are_not() -> None:
    assert A2ATimeoutError("slow").retryable is True
    assert A2AAgentNotFoundError("unknown").retryable is False
    assert A2ALoopDetectedError("cycle").retryable is False


@pytest.mark.unit
def test_bulk_index_error_retryable_and_carries_failed_items() -> None:
    err = BulkIndexError("2 failed", failed_items=[{"id": "a"}, {"id": "b"}])
    assert err.retryable is True
    assert err.failed_items == [{"id": "a"}, {"id": "b"}]


@pytest.mark.unit
def test_cache_error_retryable_by_default() -> None:
    assert CacheError("redis blip").retryable is True


@pytest.mark.unit
def test_retrieval_error_retryable_by_default() -> None:
    assert RetrievalError("transient").retryable is True


@pytest.mark.unit
def test_agent_error_not_retryable_by_default() -> None:
    assert AgentError("bad input").retryable is False


@pytest.mark.unit
def test_mcp_error_retryable_by_default() -> None:
    assert MCPError("transient").retryable is True


# --- AgentError guard clauses on agent constructors -----------------------


@pytest.mark.unit
def test_researcher_agent_rejects_none_llm() -> None:
    with pytest.raises(AgentError):
        ResearcherAgent(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_grader_agent_rejects_none_llm() -> None:
    with pytest.raises(AgentError):
        GraderAgent(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_synthesizer_agent_rejects_none_llm() -> None:
    with pytest.raises(AgentError):
        SynthesizerAgent(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_retriever_agent_rejects_none_retrieval_service() -> None:
    with pytest.raises(AgentError):
        RetrieverAgent(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_citation_agent_rejects_none_llm() -> None:
    with pytest.raises(AgentError):
        CitationAgent(None)  # type: ignore[arg-type]


@pytest.mark.unit
def test_agent_constructors_accept_valid_dependencies() -> None:
    llm = FakeLLM()
    ResearcherAgent(llm)
    GraderAgent(llm)
    SynthesizerAgent(llm)
    RetrieverAgent(object())  # type: ignore[arg-type]
    CitationAgent(llm)  # now requires an llm for sentence-to-chunk mapping


# --- MCPError wraps unexpected agent failures at the MCP boundary ---------


@pytest.mark.unit
async def test_mcp_server_wraps_unexpected_agent_failure() -> None:
    from fastmcp import Client

    from querymesh.mcp.servers import build_research_server

    class _BrokenLLM:
        model = "broken"

        async def ainvoke(self, prompt: str):
            raise RuntimeError("provider is down")

    server = build_research_server(ResearcherAgent(_BrokenLLM()))  # type: ignore[arg-type]

    async with Client(server) as client:
        result = await client.call_tool("search_papers", {"query": "q"}, raise_on_error=False)

    assert result.is_error is True


@pytest.mark.unit
async def test_mcp_server_succeeds_normally_without_wrapping_interfering() -> None:
    from fastmcp import Client

    from querymesh.mcp.servers import build_research_server

    llm = FakeLLM(response_text=json.dumps({"search_queries": ["q1"]}))
    server = build_research_server(ResearcherAgent(llm))

    async with Client(server) as client:
        result = await client.call_tool("search_papers", {"query": "q"})

    assert result.data["search_queries"] == ["q1"]


# --- RetrievalService wraps unexpected retriever failures ------------------


@pytest.mark.unit
async def test_retrieval_service_wraps_unexpected_failure() -> None:
    from querymesh.retrieval.service import RetrievalService

    class _BrokenHybridRetriever:
        async def search(self, query, top_k=10, filters=None):
            raise RuntimeError("cluster unreachable")

    service = RetrievalService(hybrid_retriever=_BrokenHybridRetriever())  # type: ignore[arg-type]

    with pytest.raises(RetrievalError) as exc_info:
        await service.retrieve("q")

    assert exc_info.value.retryable is True


@pytest.mark.unit
async def test_retrieval_service_does_not_rewrap_query_mesh_errors() -> None:
    from querymesh.retrieval.service import RetrievalService

    class _RetrieverRaisingTypedError:
        async def search(self, query, top_k=10, filters=None):
            raise IndexingError("already well-typed")

    service = RetrievalService(hybrid_retriever=_RetrieverRaisingTypedError())  # type: ignore[arg-type]

    with pytest.raises(IndexingError):  # not wrapped in a generic RetrievalError
        await service.retrieve("q")
