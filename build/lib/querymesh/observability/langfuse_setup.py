"""Langfuse configuration.

Same optional-fallback pattern as `logfire_setup.py`: no-op when
credentials aren't configured.
"""

from __future__ import annotations

from typing import Any

from querymesh.config import Settings, get_settings
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


def configure_langfuse(settings: Settings | None = None) -> Any | None:
    """Construct and return a `Langfuse` client, or None if not configured."""
    settings = settings or get_settings()

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("Langfuse credentials are set but the 'langfuse' package is not installed")
        return None

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
