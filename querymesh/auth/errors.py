"""Auth-layer errors.

Both descend from `QueryMeshError` (Phase 23). Neither is retryable — a
request with a missing/expired/invalid token or insufficient scopes will
fail identically on retry; the caller must get a new/different token, not
try again.
"""

from __future__ import annotations

from querymesh.errors import QueryMeshError


class AuthenticationError(QueryMeshError):
    """The request has no valid, verifiable identity: missing, malformed,
    expired, or bad-signature token. Maps to HTTP 401.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class AuthorizationError(QueryMeshError):
    """The request has a valid identity but lacks permission for what it's
    asking to do. Maps to HTTP 403.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
