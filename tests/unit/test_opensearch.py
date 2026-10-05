"""Phase 5 tests: OpenSearchConnection against a fake, in-memory client
implementing the `OpenSearchClientProtocol` surface — no live cluster
required. `build_client()` (the real `AsyncOpenSearch` constructor path)
is tested separately for its configuration-validation behavior only.
"""

from __future__ import annotations

from typing import Any

import pytest

from querymesh.config import Settings
from querymesh.indexing import (
    BulkIndexError,
    IndexingError,
    OpenSearchConnection,
    build_client,
    build_index_settings,
)


class _FakeIndicesClient:
    def __init__(self) -> None:
        self._indices: dict[str, dict] = {}

    async def exists(self, index: str) -> bool:
        return index in self._indices

    async def create(self, index: str, body: dict) -> dict:
        self._indices[index] = body
        return {"acknowledged": True}

    async def delete(self, index: str) -> dict:
        self._indices.pop(index, None)
        return {"acknowledged": True}


class _FakeOpenSearchClient:
    """In-memory double implementing the subset of AsyncOpenSearch used by
    OpenSearchConnection.
    """

    def __init__(self) -> None:
        self.indices = _FakeIndicesClient()
        self._docs: dict[str, dict[str, dict]] = {}
        self.closed = False
        self.ping_result = True
        self.search_calls: list[dict] = []

    async def ping(self) -> bool:
        return self.ping_result

    async def index(self, *, index: str, id: str, body: dict) -> dict:
        self._docs.setdefault(index, {})[id] = body
        return {"result": "created", "_id": id}

    async def get(self, *, index: str, id: str) -> dict:
        return {"_id": id, "_source": self._docs[index][id]}

    async def delete(self, *, index: str, id: str) -> dict:
        del self._docs[index][id]
        return {"result": "deleted"}

    async def update(self, *, index: str, id: str, body: dict) -> dict:
        self._docs[index][id].update(body["doc"])
        return {"result": "updated"}

    async def search(self, *, index: str, body: dict) -> dict:
        self.search_calls.append({"index": index, "body": body})
        docs = self._docs.get(index, {})
        hits = [{"_id": doc_id, "_source": src} for doc_id, src in docs.items()]
        return {"hits": {"total": {"value": len(hits)}, "hits": hits}}

    async def bulk(self, *, body: list) -> dict:
        errors = False
        items = []
        # body alternates {"index": {"_index":..,"_id":..}} , <source doc>
        for i in range(0, len(body), 2):
            action = body[i]["index"]
            source = body[i + 1]
            index_name, doc_id = action["_index"], action["_id"]
            if source.get("_simulate_failure"):
                errors = True
                items.append({"index": {"_id": doc_id, "status": 400, "error": "simulated failure"}})
                continue
            self._docs.setdefault(index_name, {})[doc_id] = source
            items.append({"index": {"_id": doc_id, "status": 201}})
        return {"errors": errors, "items": items}

    async def close(self) -> None:
        self.closed = True


def _connection() -> tuple[OpenSearchConnection, _FakeOpenSearchClient]:
    client = _FakeOpenSearchClient()
    return OpenSearchConnection(client=client, index_name="querymesh-test"), client


@pytest.mark.unit
async def test_health_check_true_when_ping_succeeds() -> None:
    conn, client = _connection()
    assert await conn.health_check() is True


@pytest.mark.unit
async def test_health_check_false_when_ping_raises() -> None:
    conn, client = _connection()

    async def failing_ping() -> bool:
        raise ConnectionError("cluster unreachable")

    client.ping = failing_ping  # type: ignore[method-assign]
    assert await conn.health_check() is False


@pytest.mark.unit
async def test_create_index_is_idempotent() -> None:
    conn, client = _connection()
    await conn.create_index()
    assert await conn.index_exists() is True

    # Calling again should not raise or duplicate — just skip.
    await conn.create_index()
    assert client.indices._indices.keys() == {"querymesh-test"}


@pytest.mark.unit
async def test_recreate_index_drops_and_recreates() -> None:
    conn, client = _connection()
    await conn.create_index()
    original_body = client.indices._indices["querymesh-test"]

    await conn.recreate_index(embedding_dim=256)
    new_body = client.indices._indices["querymesh-test"]

    assert new_body != original_body
    assert new_body["mappings"]["properties"]["embedding"]["dimension"] == 256


@pytest.mark.unit
async def test_index_get_update_delete_document_roundtrip() -> None:
    conn, client = _connection()
    await conn.create_index()

    await conn.index_document("2608.00001v1:chunk:0000", {"text": "hello", "chunk_index": 0})
    result = await client.get(index="querymesh-test", id="2608.00001v1:chunk:0000")
    assert result["_source"]["text"] == "hello"

    await conn.update_document("2608.00001v1:chunk:0000", {"text": "updated"})
    result = await client.get(index="querymesh-test", id="2608.00001v1:chunk:0000")
    assert result["_source"]["text"] == "updated"

    await conn.delete_document("2608.00001v1:chunk:0000")
    assert "2608.00001v1:chunk:0000" not in client._docs["querymesh-test"]


@pytest.mark.unit
async def test_bulk_index_succeeds() -> None:
    conn, client = _connection()
    await conn.create_index()

    docs = [
        {"chunk_id": "a:chunk:0000", "text": "one"},
        {"chunk_id": "a:chunk:0001", "text": "two"},
    ]
    count = await conn.bulk_index(docs)

    assert count == 2
    assert len(client._docs["querymesh-test"]) == 2


@pytest.mark.unit
async def test_bulk_index_empty_list_is_noop() -> None:
    conn, client = _connection()
    count = await conn.bulk_index([])
    assert count == 0


@pytest.mark.unit
async def test_bulk_index_raises_bulk_index_error_on_partial_failure() -> None:
    conn, client = _connection()
    await conn.create_index()

    docs = [
        {"chunk_id": "a:chunk:0000", "text": "ok"},
        {"chunk_id": "a:chunk:0001", "text": "bad", "_simulate_failure": True},
    ]

    with pytest.raises(BulkIndexError) as exc_info:
        await conn.bulk_index(docs)

    assert len(exc_info.value.failed_items) == 1


@pytest.mark.unit
async def test_search_delegates_to_client_with_index_name() -> None:
    conn, client = _connection()
    await conn.create_index()
    await conn.index_document("a:chunk:0000", {"text": "hello world"})

    result = await conn.search({"query": {"match_all": {}}})

    assert client.search_calls[-1]["index"] == "querymesh-test"
    assert result["hits"]["total"]["value"] == 1


@pytest.mark.unit
async def test_close_closes_underlying_client() -> None:
    conn, client = _connection()
    await conn.close()
    assert client.closed is True


@pytest.mark.unit
def test_build_index_settings_supports_all_field_kinds() -> None:
    body = build_index_settings(embedding_dim=512)
    props = body["mappings"]["properties"]

    assert props["title"]["type"] == "text"
    assert props["arxiv_id"]["type"] == "keyword"
    assert props["published"]["type"] == "date"
    assert props["chunk_index"]["type"] == "integer"
    assert props["embedding"]["type"] == "knn_vector"
    assert props["embedding"]["dimension"] == 512
    assert body["settings"]["index"]["knn"] is True


@pytest.mark.unit
def test_build_client_raises_when_url_not_configured() -> None:
    settings = Settings(OPENSEARCH_URL=None)
    with pytest.raises(IndexingError):
        build_client(settings)


@pytest.mark.unit
def test_build_client_constructs_real_client_when_configured() -> None:
    settings = Settings(OPENSEARCH_URL="http://localhost:9200")
    client = build_client(settings)
    # We don't connect — just verify the real AsyncOpenSearch client was built.
    from opensearchpy import AsyncOpenSearch

    assert isinstance(client, AsyncOpenSearch)
