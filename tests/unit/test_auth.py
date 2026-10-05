"""Phase 32 unit tests: JWTValidator (signature/expiry/issuer/audience
validation), issue_dev_token, and the auth error hierarchy — no FastAPI
involved, just the JWT logic in isolation.
"""

from __future__ import annotations

import time

import jwt as pyjwt
import pytest
from querymesh.auth import AuthenticationError, JWTValidator, TokenClaims, issue_dev_token
from querymesh.config import Settings
from querymesh.errors import ConfigurationError, QueryMeshError


def _settings(**overrides) -> Settings:
    defaults = dict(JWT_SECRET_KEY="test-secret-key-at-least-32-bytes-long-for-hs256")
    defaults.update(overrides)
    return Settings(**defaults)


# --- Construction -----------------------------------------------------


@pytest.mark.unit
def test_validator_requires_a_key_configured() -> None:
    settings = Settings(JWT_SECRET_KEY=None, JWT_PUBLIC_KEY=None)
    with pytest.raises(ConfigurationError):
        JWTValidator(settings)


@pytest.mark.unit
def test_validator_constructs_with_symmetric_key() -> None:
    validator = JWTValidator(_settings())
    assert validator is not None


@pytest.mark.unit
def test_validator_constructs_with_asymmetric_key() -> None:
    settings = Settings(JWT_SECRET_KEY=None, JWT_PUBLIC_KEY="fake-public-key", JWT_ALGORITHM="RS256")
    validator = JWTValidator(settings)
    assert validator is not None


# --- issue_dev_token ----------------------------------------------------


@pytest.mark.unit
def test_issue_dev_token_requires_secret_key() -> None:
    settings = Settings(JWT_SECRET_KEY=None)
    with pytest.raises(ConfigurationError):
        issue_dev_token("user-1", settings=settings)


@pytest.mark.unit
def test_issue_dev_token_produces_decodable_token() -> None:
    settings = _settings()
    token = issue_dev_token("user-1", scopes=["query:submit"], settings=settings)

    validator = JWTValidator(settings)
    claims = validator.decode(token)

    assert claims.subject == "user-1"
    assert claims.scopes == ["query:submit"]


# --- Valid token decode --------------------------------------------------


@pytest.mark.unit
def test_decode_valid_token_returns_claims() -> None:
    settings = _settings()
    token = issue_dev_token("alice", scopes=["query:submit", "admin"], settings=settings)

    claims = JWTValidator(settings).decode(token)

    assert isinstance(claims, TokenClaims)
    assert claims.subject == "alice"
    assert claims.has_scope("query:submit")
    assert claims.has_scope("admin")
    assert not claims.has_scope("nonexistent")
    assert claims.expires_at is not None


@pytest.mark.unit
def test_decode_token_with_no_scopes() -> None:
    settings = _settings()
    token = issue_dev_token("bob", settings=settings)

    claims = JWTValidator(settings).decode(token)

    assert claims.scopes == []
    assert claims.has_scope("anything") is False


# --- Missing token -----------------------------------------------------


@pytest.mark.unit
def test_decode_empty_token_raises_authentication_error() -> None:
    validator = JWTValidator(_settings())
    with pytest.raises(AuthenticationError):
        validator.decode("")


# --- Malformed token -----------------------------------------------------


@pytest.mark.unit
def test_decode_malformed_token_raises_authentication_error() -> None:
    validator = JWTValidator(_settings())
    with pytest.raises(AuthenticationError):
        validator.decode("not.a.valid.jwt.at.all")


# --- Expired token -----------------------------------------------------


@pytest.mark.unit
def test_decode_expired_token_raises_authentication_error() -> None:
    settings = _settings()
    now = int(time.time())
    expired_token = pyjwt.encode(
        {"sub": "carol", "exp": now - 3600, "iat": now - 7200},
        key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    validator = JWTValidator(settings)
    with pytest.raises(AuthenticationError, match="expired"):
        validator.decode(expired_token)


# --- Invalid signature -----------------------------------------------------


@pytest.mark.unit
def test_decode_wrong_signature_raises_authentication_error() -> None:
    token = issue_dev_token("dave", settings=_settings(JWT_SECRET_KEY="secret-a-at-least-32-bytes-long!"))

    validator = JWTValidator(_settings(JWT_SECRET_KEY="secret-b-at-least-32-bytes-long!"))
    with pytest.raises(AuthenticationError):
        validator.decode(token)


# --- Issuer / audience validation ----------------------------------------


@pytest.mark.unit
def test_decode_rejects_wrong_issuer() -> None:
    settings = _settings(JWT_ISSUER="expected-issuer")
    now = int(time.time())
    token = pyjwt.encode(
        {"sub": "eve", "exp": now + 3600, "iss": "wrong-issuer"},
        key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AuthenticationError):
        JWTValidator(settings).decode(token)


@pytest.mark.unit
def test_decode_accepts_matching_issuer() -> None:
    settings = _settings(JWT_ISSUER="my-issuer")
    token = issue_dev_token("frank", settings=settings)

    claims = JWTValidator(settings).decode(token)
    assert claims.issuer == "my-issuer"


@pytest.mark.unit
def test_decode_rejects_wrong_audience() -> None:
    settings = _settings(JWT_AUDIENCE="expected-aud")
    now = int(time.time())
    token = pyjwt.encode(
        {"sub": "grace", "exp": now + 3600, "aud": "wrong-aud"},
        key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(AuthenticationError):
        JWTValidator(settings).decode(token)


# --- Missing required claims -----------------------------------------------


@pytest.mark.unit
def test_decode_rejects_token_missing_subject() -> None:
    settings = _settings()
    now = int(time.time())
    token = pyjwt.encode({"exp": now + 3600}, key=settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    with pytest.raises(AuthenticationError):
        JWTValidator(settings).decode(token)


@pytest.mark.unit
def test_decode_rejects_token_missing_expiry() -> None:
    settings = _settings()
    token = pyjwt.encode({"sub": "henry"}, key=settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    with pytest.raises(AuthenticationError):
        JWTValidator(settings).decode(token)


# --- scope() / scope string fallback --------------------------------------


@pytest.mark.unit
def test_decode_reads_space_separated_scope_string() -> None:
    """Some IdPs (OAuth2-style) use a single space-separated `scope`
    string claim rather than a `scopes` list — support both shapes.
    """
    settings = _settings()
    now = int(time.time())
    token = pyjwt.encode(
        {"sub": "ivan", "exp": now + 3600, "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience, "scope": "read write admin"},
        key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    claims = JWTValidator(settings).decode(token)
    assert claims.scopes == ["read", "write", "admin"]


# --- Error hierarchy -----------------------------------------------------


@pytest.mark.unit
def test_authentication_error_never_retryable() -> None:
    from querymesh.auth import AuthenticationError

    assert AuthenticationError("bad token").retryable is False


@pytest.mark.unit
def test_authorization_error_never_retryable() -> None:
    from querymesh.auth import AuthorizationError

    assert AuthorizationError("insufficient scope").retryable is False


@pytest.mark.unit
def test_auth_errors_descend_from_query_mesh_error() -> None:
    from querymesh.auth import AuthenticationError, AuthorizationError

    assert issubclass(AuthenticationError, QueryMeshError)
    assert issubclass(AuthorizationError, QueryMeshError)


# --- require_scopes / get_current_user dependency logic --------------------


@pytest.mark.unit
async def test_require_scopes_passes_with_all_required_scopes() -> None:
    from querymesh.auth.dependencies import require_scopes

    claims = TokenClaims(subject="jill", scopes=["query:submit", "admin"])
    inner = require_scopes("query:submit")

    result = await inner(user=claims)
    assert result is claims


@pytest.mark.unit
async def test_require_scopes_raises_when_missing_a_scope() -> None:
    from querymesh.auth.dependencies import require_scopes
    from querymesh.auth.errors import AuthorizationError

    claims = TokenClaims(subject="jack", scopes=["read"])
    inner = require_scopes("query:submit")

    with pytest.raises(AuthorizationError):
        await inner(user=claims)


@pytest.mark.unit
async def test_require_scopes_requires_all_listed_scopes() -> None:
    from querymesh.auth.dependencies import require_scopes
    from querymesh.auth.errors import AuthorizationError

    claims = TokenClaims(subject="kate", scopes=["query:submit"])  # missing "admin"
    inner = require_scopes("query:submit", "admin")

    with pytest.raises(AuthorizationError):
        await inner(user=claims)
