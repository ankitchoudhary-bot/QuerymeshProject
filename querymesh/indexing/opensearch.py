"""Thin async wrapper around an OpenSearch client.

The real client (`opensearchpy.AsyncOpenSearch`) is constructed by
`build_client()` from `Settings`, but `OpenSearchConnection` accepts *any*
object exposing the same handful of methods we actually use — this is
what makes it testable without a live cluster (see
`tests/unit/test_opensearch.py`, which injects a hand-built fake client).
"""

from __future__ import annotations

from typing import Any, Protocol

from querymesh.config import Settings, get_settings
from querymesh.indexing.errors import BulkIndexError, IndexingError
from querymesh.indexing.mappings import build_index_settings
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


class OpenSearchClientProtocol(Protocol):
    """The subset of `AsyncOpenSearch`'s surface this wrapper depends on."""

    indices: Any

    async def ping(self, **kwargs: Any) -> bool: ...
    async def index(self, *, index: str, body: dict, id: str | None = None, **kwargs: Any) -> dict: ...
    async def get(self, *, index: str, id: str, **kwargs: Any) -> dict: ...
    async def delete(self, *, index: str, id: str, **kwargs: Any) -> dict: ...
    async def update(self, *, index: str, id: str, body: dict, **kwargs: Any) -> dict: ...
    async def search(self, *, index: str, body: dict, **kwargs: Any) -> dict: ...
    async def bulk(self, *, body: list, **kwargs: Any) -> dict: ...
    async def close(self) -> None: ...


def build_client(settings: Settings | None = None):
    """Construct a real `AsyncOpenSearch` client from Settings.

    Imported lazily by callers that want the real thing; tests construct
    `OpenSearchConnection` directly with a fake client instead of calling
    this.
    """
    from opensearchpy import AsyncOpenSearch

    settings = settings or get_settings()
    if not settings.opensearch_url:
        raise IndexingError("OPENSEARCH_URL is not configured")

    http_auth = None
    if settings.opensearch_username and settings.opensearch_password:
        http_auth = (settings.opensearch_username, settings.opensearch_password)

    return AsyncOpenSearch(hosts=[settings.opensearch_url], http_auth=http_auth,  use_ssl=True, verify_certs=False, )


class OpenSearchConnection:
    """CRUD + bulk + admin operations against one OpenSearch index."""

    def __init__(self, client: OpenSearchClientProtocol, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    async def health_check(self) -> bool:
        try:
            return bool(await self.client.ping())
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenSearch health check failed: %s", exc)
            return False

    async def index_exists(self) -> bool:
        return bool(await self.client.indices.exists(index=self.index_name))

    async def create_index(self, embedding_dim: int | None = None) -> None:
        if await self.index_exists():
            logger.info("Index %s already exists, skipping creation", self.index_name)
            return
        body = build_index_settings(embedding_dim) if embedding_dim else build_index_settings()
        await self.client.indices.create(index=self.index_name, body=body)
        logger.info("Created index %s", self.index_name)

    async def recreate_index(self, embedding_dim: int | None = None) -> None:
        if await self.index_exists():
            await self.client.indices.delete(index=self.index_name)
            logger.info("Deleted existing index %s", self.index_name)
        await self.create_index(embedding_dim=embedding_dim)

    async def index_document(self, doc_id: str, document: dict) -> None:
        await self.client.index(index=self.index_name, id=doc_id, body=document)

    async def bulk_index(self, documents: list[dict], id_field: str = "chunk_id") -> int:
        """Bulk-index documents. Returns the number successfully indexed.

        Raises `BulkIndexError` (carrying the failed items) if any document
        fails — callers that want best-effort behavior can catch it and
        inspect `.failed_items`.
        """
        if not documents:
            return 0

        actions = [
            {
                "_index": self.index_name,
                "_id": doc[id_field],
                "_source": doc,
            }
            for doc in documents
        ]

        response = await self.client.bulk(body=_actions_to_bulk_body(actions))

        failed_items = []
        succeeded = 0
        if response.get("errors"):
            for item in response.get("items", []):
                # Each item is {"index": {...}} (or "create"/"update"/"delete").
                op_result = next(iter(item.values()))
                if op_result.get("status", 200) >= 300 or "error" in op_result:
                    failed_items.append(op_result)
                else:
                    succeeded += 1
            raise BulkIndexError(
                f"{len(failed_items)} of {len(documents)} document(s) failed during bulk indexing",
                failed_items=failed_items,
            )

        return len(documents)

    async def delete_document(self, doc_id: str) -> None:
        await self.client.delete(index=self.index_name, id=doc_id)

    async def update_document(self, doc_id: str, partial_document: dict) -> None:
        await self.client.update(index=self.index_name, id=doc_id, body={"doc": partial_document})

    async def search(self, body: dict) -> dict:
        return await self.client.search(index=self.index_name, body=body)

    async def close(self) -> None:
        await self.client.close()


def _actions_to_bulk_body(actions: list[dict]) -> list[dict]:
    """Expand `{_index, _id, _source}` actions into the OpenSearch bulk
    API's alternating action/document line format.
    """
    body: list[dict] = []
    for action in actions:
        body.append({"index": {"_index": action["_index"], "_id": action["_id"]}})
        body.append(action["_source"])
    return body
