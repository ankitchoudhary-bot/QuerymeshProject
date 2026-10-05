"""QueryMesh's project-wide exception hierarchy.

Every custom error in the codebase descends from `QueryMeshError`. Each
carries a `retryable: bool` so callers (retry loops, the graph's node
error-isolation, A2A dispatch) can decide programmatically whether a
failure is worth retrying rather than pattern-matching on error text.

Module-specific error files (`ingestion/errors.py`, `embeddings/base.py`,
`llm/base.py`, `indexing/errors.py`, `a2a/errors.py`, `cache/errors.py`)
define the concrete errors used day-to-day; this module is where they all
plug into the shared hierarchy, plus the errors that don't have a more
specific home (`ConfigurationError`, `RetrievalError`, `AgentError`,
`MCPError`).
"""

from __future__ import annotations


class QueryMeshError(Exception):
    """Base for all QueryMesh errors."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ConfigurationError(QueryMeshError):
    """Invalid or missing configuration. Never retryable — retrying with
    the same bad config just fails again; the config itself must change.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class RetrievalError(QueryMeshError):
    """Failure in the retrieval layer (BM25/vector/hybrid/service) not
    already covered by a more specific error type. Retryable by default —
    most retrieval failures are transient (cluster hiccup, timeout).
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class AgentError(QueryMeshError):
    """An agent received invalid input or hit an unrecoverable internal
    error (as opposed to a gracefully-handled LLM parsing failure, which
    agents already degrade from without raising).
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)


class MCPError(QueryMeshError):
    """An MCP tool adapter failed to complete a call to its underlying
    agent. Retryable by default — mirrors the transport-level retry
    behavior MCP clients typically apply.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class ValidationError(QueryMeshError):
    """Input or output failed a production safety limit (max query
    length, max document count, etc). Never retryable — the caller must
    change the input, not simply try again.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
