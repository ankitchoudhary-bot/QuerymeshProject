"""FastAPI auth dependencies.

`get_current_user` authenticates (valid token → identity); `require_scopes`
additionally authorizes (identity → permission for this specific route).
Kept as a separate, composable dependency rather than folded into
`require_scopes` alone, so a route that only needs "some valid caller,
any scope" doesn't have to invent an empty scope list to express that.
"""

from __future__ import annotations

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from querymesh.auth.errors import AuthenticationError, AuthorizationError
from querymesh.auth.jwt_provider import JWTValidator
from querymesh.auth.models import TokenClaims

# auto_error=False: we want to raise our own AuthenticationError (which our
# registered exception handler turns into a structured ErrorResponse) rather
# than FastAPI's default plain-text 403 for a missing Authorization header.
_bearer_scheme = HTTPBearer(auto_error=False)


def get_jwt_validator() -> JWTValidator:
    """Constructed fresh per call (cheap — just reads Settings) rather than
    cached at import time, so tests can reconfigure JWT settings between
    calls without needing to reset module-level state.
    """
    return JWTValidator()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    validator: JWTValidator = Depends(get_jwt_validator),
) -> TokenClaims:
    if credentials is None:
        raise AuthenticationError("Missing Authorization header (expected: Bearer <token>)")
    return validator.decode(credentials.credentials)


def require_scopes(*required_scopes: str):
    """Dependency factory: `Depends(require_scopes("query:submit"))`.

    Requires ALL listed scopes to be present (AND, not OR) — routes
    needing "any of" a set of scopes should compose their own check
    instead of assuming that semantic here.
    """

    async def _check(user: TokenClaims = Depends(get_current_user)) -> TokenClaims:
        missing = [s for s in required_scopes if not user.has_scope(s)]
        if missing:
            raise AuthorizationError(f"Missing required scope(s): {', '.join(missing)}")
        return user

    return _check
