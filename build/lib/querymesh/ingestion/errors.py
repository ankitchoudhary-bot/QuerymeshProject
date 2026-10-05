"""Ingestion-layer errors.

Both descend from `QueryMeshError` (Phase 23), so callers can catch either
`IngestionError` specifically or `QueryMeshError` generically, and can
check `.retryable` rather than pattern-matching on message text.
"""

from __future__ import annotations

from querymesh.errors import QueryMeshError


class IngestionError(QueryMeshError):
    """Base error for ingestion-layer failures. Retryable by default —
    most ingestion failures (network, transient parsing issues) are.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class ArxivError(IngestionError):
    """Raised when the arXiv API cannot be queried or returns bad data."""
