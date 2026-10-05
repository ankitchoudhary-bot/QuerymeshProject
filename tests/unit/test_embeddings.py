"""Phase 7 tests: FakeEmbedder, JinaEmbedder (batching/retries/timeout/
dimension validation via httpx.MockTransport), and the embedder factory.
"""

from __future__ import annotations

import httpx
import pytest

from querymesh.config import Settings
from querymesh.embeddings import (
    EmbeddingDimensionError,
    EmbeddingError,
    FakeEmbedder,
    JinaEmbedder,
    get_embedder,
)


# --- FakeEmbedder --------------------------------------------------------


@pytest.mark.unit
async def test_fake_embedder_is_deterministic() -> None:
    embedder = FakeEmbedder(dimension=32)
    v1 = await embedder.embed_query("hello world")
    v2 = await embedder.embed_query("hello world")
    assert v1 == v2
    assert len(v1) == 32


@pytest.mark.unit
async def test_fake_embedder_different_texts_differ() -> None:
    embedder = FakeEmbedder(dimension=32)
    v1 = await embedder.embed_query("alpha")
    v2 = await embedder.embed_query("beta")
    assert v1 != v2


@pytest.mark.unit
async def test_fake_embedder_embed_documents_batch() -> None:
    embedder = FakeEmbedder(dimension=16)
    vectors = await embedder.embed_documents(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == 16 for v in vectors)


# --- JinaEmbedder ----------------------------------------------------------


def _jina_response(embeddings: list[list[float]]) -> dict:
    return {
        "model": "jina-embeddings-v3",
        "data": [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings)],
        "usage": {"total_tokens": 10},
    }


@pytest.mark.unit
def test_jina_embedder_requires_api_key() -> None:
    with pytest.raises(EmbeddingError):
        JinaEmbedder(api_key="", model="jina-embeddings-v3")


@pytest.mark.unit
def test_jina_embedder_requires_model() -> None:
    with pytest.raises(EmbeddingError):
        JinaEmbedder(api_key="secret", model="")


@pytest.mark.unit
async def test_jina_embed_query_sends_auth_header_and_task() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_jina_response([[0.1, 0.2, 0.3]]))

    embedder = JinaEmbedder(
        api_key="secret-key",
        model="jina-embeddings-v3",
        transport=httpx.MockTransport(handler),
    )
    try:
        vector = await embedder.embed_query("what is attention")
    finally:
        await embedder.aclose()

    assert vector == [0.1, 0.2, 0.3]
    assert captured["auth"] == "Bearer secret-key"
    assert captured["body"]["task"] == "retrieval.query"
    assert captured["body"]["input"] == ["what is attention"]


@pytest.mark.unit
async def test_jina_embed_documents_uses_passage_task() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        assert body["task"] == "retrieval.passage"
        return httpx.Response(200, json=_jina_response([[0.1] * 4, [0.2] * 4]))

    embedder = JinaEmbedder(
        api_key="secret-key", model="jina-embeddings-v3", transport=httpx.MockTransport(handler)
    )
    try:
        vectors = await embedder.embed_documents(["doc one", "doc two"])
    finally:
        await embedder.aclose()

    assert len(vectors) == 2
    assert vectors[0] == [0.1] * 4


@pytest.mark.unit
async def test_jina_embed_documents_batches_requests() -> None:
    call_sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        call_sizes.append(len(body["input"]))
        return httpx.Response(200, json=_jina_response([[0.0] * 4 for _ in body["input"]]))

    embedder = JinaEmbedder(
        api_key="secret-key",
        model="jina-embeddings-v3",
        batch_size=2,
        transport=httpx.MockTransport(handler),
    )
    try:
        vectors = await embedder.embed_documents(["a", "b", "c", "d", "e"])
    finally:
        await embedder.aclose()

    assert len(vectors) == 5
    assert call_sizes == [2, 2, 1]


@pytest.mark.unit
async def test_jina_embed_documents_empty_list_makes_no_request() -> None:
    called = {"yes": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["yes"] = True
        return httpx.Response(200, json=_jina_response([]))

    embedder = JinaEmbedder(
        api_key="secret-key", model="jina-embeddings-v3", transport=httpx.MockTransport(handler)
    )
    try:
        vectors = await embedder.embed_documents([])
    finally:
        await embedder.aclose()

    assert vectors == []
    assert called["yes"] is False


@pytest.mark.unit
async def test_jina_embedder_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(500)
        return httpx.Response(200, json=_jina_response([[0.1, 0.2]]))

    embedder = JinaEmbedder(
        api_key="secret-key", model="jina-embeddings-v3", transport=httpx.MockTransport(handler)
    )
    try:
        vector = await embedder.embed_query("retry test")
    finally:
        await embedder.aclose()

    assert attempts["count"] == 3
    assert vector == [0.1, 0.2]


@pytest.mark.unit
async def test_jina_embedder_raises_after_exhausting_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    embedder = JinaEmbedder(
        api_key="secret-key",
        model="jina-embeddings-v3",
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )
    try:
        with pytest.raises(EmbeddingError):
            await embedder.embed_query("always fails")
    finally:
        await embedder.aclose()


@pytest.mark.unit
async def test_jina_embedder_validates_expected_dimension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_jina_response([[0.1, 0.2, 0.3]]))  # dim=3

    embedder = JinaEmbedder(
        api_key="secret-key",
        model="jina-embeddings-v3",
        expected_dimension=1024,
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(EmbeddingDimensionError):
            await embedder.embed_query("dimension mismatch")
    finally:
        await embedder.aclose()


@pytest.mark.unit
async def test_jina_embedder_rejects_inconsistent_batch_dimensions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_jina_response([[0.1, 0.2], [0.1, 0.2, 0.3]]))

    embedder = JinaEmbedder(
        api_key="secret-key", model="jina-embeddings-v3", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(EmbeddingDimensionError):
            await embedder.embed_documents(["a", "b"])
    finally:
        await embedder.aclose()


@pytest.mark.unit
async def test_jina_embedder_malformed_response_raises_embedding_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    embedder = JinaEmbedder(
        api_key="secret-key", model="jina-embeddings-v3", transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(EmbeddingError):
            await embedder.embed_query("bad response")
    finally:
        await embedder.aclose()


# --- Factory ------------------------------------------------------------


@pytest.mark.unit
def test_factory_falls_back_to_fake_without_api_key() -> None:
    settings = Settings(JINA_API_KEY=None)
    embedder = get_embedder(settings)
    assert isinstance(embedder, FakeEmbedder)


@pytest.mark.unit
def test_factory_requires_model_when_api_key_set() -> None:
    settings = Settings(JINA_API_KEY="secret", JINA_EMBEDDING_MODEL=None)
    with pytest.raises(EmbeddingError):
        get_embedder(settings)


@pytest.mark.unit
def test_factory_constructs_jina_embedder_when_configured() -> None:
    settings = Settings(JINA_API_KEY="secret", JINA_EMBEDDING_MODEL="jina-embeddings-v3")
    embedder = get_embedder(settings)
    assert isinstance(embedder, JinaEmbedder)
    assert embedder.model == "jina-embeddings-v3"
