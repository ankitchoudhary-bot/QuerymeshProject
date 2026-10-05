"""A2A-layer errors."""

from __future__ import annotations

from querymesh.errors import QueryMeshError


class A2AError(QueryMeshError):
    """Base error for agent-to-agent communication failures."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class A2ATimeoutError(A2AError):
    """Raised when a recipient doesn't respond within the configured
    timeout. Retryable — the recipient may simply be slow this time.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class A2AAgentNotFoundError(A2AError):
    """Raised when sending to a recipient with no registered handler.
    Never retryable — the recipient won't suddenly exist on retry.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class A2ALoopDetectedError(A2AError):
    """Raised when a message's hop count exceeds the configured maximum,
    indicating a delegation cycle rather than forward progress. Never
    retryable — retrying repeats the same cycle.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
