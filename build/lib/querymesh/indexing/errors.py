"""Indexing-layer errors."""

from __future__ import annotations

from querymesh.errors import QueryMeshError


class IndexingError(QueryMeshError):
    """Base error for OpenSearch indexing-layer failures. Retryable by
    default — most indexing failures (cluster hiccup, timeout) are.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class ConnectionError_(IndexingError):
    """Raised when the OpenSearch cluster cannot be reached."""


class BulkIndexError(IndexingError):
    """Raised when one or more documents fail during a bulk indexing call."""

    def __init__(self, message: str, failed_items: list[dict] | None = None) -> None:
        super().__init__(message, retryable=True)
        self.failed_items = failed_items or []
