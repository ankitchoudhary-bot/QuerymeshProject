"""MCP authentication strategy: static bearer token verification.

QueryMesh's FastMCP servers (Phase 16) are meant to run inside a trusted
network (internal service mesh, same-host process, or behind a gateway
that already authenticates callers) — so a lightweight static bearer
token, not a full OAuth flow, is the appropriate strategy here. If a
stronger requirement shows up later (multi-tenant, external exposure),
FastMCP's OAuth providers (`fastmcp.server.auth.providers.*`) are a
drop-in replacement for this `AuthProvider` without changing how servers
are built.

Servers are unauthenticated (`build_..._server(agent)`, no token) unless
explicitly given one, matching every other optional-security-hardening
default in this project (e.g. Redis/OpenSearch have no auth by default
either, in `.env.example`).
"""

from __future__ import annotations

import hmac

from fastmcp.server.auth.auth import AccessToken, TokenVerifier

from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


class StaticBearerTokenVerifier(TokenVerifier):
    """Verifies a single, pre-shared bearer token.

    Uses `hmac.compare_digest` for the comparison — a plain `==` on
    secret tokens is a timing side-channel; this is the standard fix.
    """

    def __init__(self, expected_token: str) -> None:
        if not expected_token:
            raise ValueError("StaticBearerTokenVerifier requires a non-empty expected_token")
        super().__init__()
        self._expected_token = expected_token

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or not hmac.compare_digest(token, self._expected_token):
            logger.warning("Rejected MCP request with invalid bearer token")
            return None
        return AccessToken(token=token, client_id="querymesh", scopes=[])
