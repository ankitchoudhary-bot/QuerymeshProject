"""JWT validation.

QueryMesh only ever *verifies* tokens issued elsewhere (an IdP, or its own
`issue_dev_token` for local development) — it never has an authoritative
signing role in production. Symmetric (HS256, `JWT_SECRET_KEY`) and
asymmetric (RS256/ES256/..., `JWT_PUBLIC_KEY`) verification are both
supported; which one is active is entirely a matter of which key is
configured, not a code branch a caller has to choose.

Uses PyJWT (2.13.0) directly rather than a homemade implementation —
signature verification, expiry, issuer, and audience checks are all
delegated to PyJWT's own well-tested logic; this module's job is only to
translate `Settings` into `jwt.decode()` arguments and translate PyJWT's
exceptions into QueryMesh's own `AuthenticationError`.
"""

from __future__ import annotations

import time

import jwt

from querymesh.auth.errors import AuthenticationError
from querymesh.auth.models import TokenClaims
from querymesh.config import Settings, get_settings
from querymesh.errors import ConfigurationError
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_DEV_TOKEN_TTL_SECONDS = 3600


class JWTValidator:
    """Verifies bearer tokens against the configured key/algorithm."""

    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()

        key = settings.jwt_public_key or settings.jwt_secret_key
        if not key:
            raise ConfigurationError(
                "JWT verification is not configured: set JWT_SECRET_KEY (symmetric) "
                "or JWT_PUBLIC_KEY (asymmetric) to enable authentication."
            )

        self._key = key
        self._algorithm = settings.jwt_algorithm
        self._issuer = settings.jwt_issuer
        self._audience = settings.jwt_audience

    def decode(self, token: str) -> TokenClaims:
        """Verify a bearer token's signature, expiry, issuer, and audience.

        Raises `AuthenticationError` for every way a token can be invalid
        — expired, malformed, wrong signature, wrong issuer/audience,
        missing required claims — so callers only need to catch one type.
        """
        if not token:
            raise AuthenticationError("Missing bearer token")

        try:
            payload = jwt.decode(
                token,
                key=self._key,
                algorithms=[self._algorithm],
                issuer=self._issuer,
                audience=self._audience,
                options={"require": ["exp", "sub"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("Token has expired") from exc
        except jwt.MissingRequiredClaimError as exc:
            raise AuthenticationError(f"Token is missing a required claim: {exc}") from exc
        except jwt.InvalidTokenError as exc:
            # Covers InvalidSignatureError, InvalidAudienceError, InvalidIssuerError,
            # DecodeError (malformed), and anything else PyJWT considers invalid —
            # deliberately not distinguished further: the client-facing behavior
            # (reject with 401) is identical either way, and the specific reason
            # is logged server-side, not exposed to the caller.
            logger.warning("Rejected JWT: %s", exc)
            raise AuthenticationError(f"Invalid token: {exc}") from exc

        scopes = payload.get("scopes")
        if scopes is None:
            scope_str = payload.get("scope", "")
            scopes = scope_str.split() if isinstance(scope_str, str) else []

        return TokenClaims(
            subject=payload["sub"],
            scopes=list(scopes),
            issuer=payload.get("iss"),
            audience=payload.get("aud"),
            expires_at=payload.get("exp"),
            raw=payload,
        )


def issue_dev_token(
    subject: str,
    scopes: list[str] | None = None,
    expires_in_seconds: int = _DEFAULT_DEV_TOKEN_TTL_SECONDS,
    settings: Settings | None = None,
) -> str:
    """Issue a token signed with the configured dev secret, for local
    testing only. Requires `JWT_SECRET_KEY` (symmetric) — there's no
    analogous helper for asymmetric keys, since QueryMesh is never meant
    to hold a private signing key in that mode (an external IdP does).
    """
    settings = settings or get_settings()
    if not settings.jwt_secret_key:
        raise ConfigurationError(
            "issue_dev_token requires JWT_SECRET_KEY to be set (symmetric signing only)"
        )

    now = int(time.time())
    payload: dict = {
        "sub": subject,
        "scope": " ".join(scopes or []),
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    if settings.jwt_issuer:
        payload["iss"] = settings.jwt_issuer
    if settings.jwt_audience:
        payload["aud"] = settings.jwt_audience

    return jwt.encode(payload, key=settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
