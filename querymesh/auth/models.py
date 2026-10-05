"""Auth data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenClaims(BaseModel):
    """The subset of a validated JWT's claims QueryMesh actually uses.

    `raw` keeps the full decoded payload for anything not modeled here
    (e.g. custom claims a specific IdP adds) without forcing every new
    claim through a schema change.
    """

    subject: str
    scopes: list[str] = Field(default_factory=list)
    issuer: str | None = None
    audience: str | None = None
    expires_at: int | None = None
    raw: dict = Field(default_factory=dict)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes
