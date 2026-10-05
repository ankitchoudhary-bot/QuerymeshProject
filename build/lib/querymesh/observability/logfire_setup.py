"""Logfire configuration.

`configure_logfire` is a no-op (returns None) when `LOGFIRE_TOKEN` isn't
set, so the app runs fine without it — matches the pattern used for LLM/
embedding providers (Phase 1/7): optional, falls back safely.
"""

from __future__ import annotations

from typing import Any

from querymesh.config import Settings, get_settings
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


def configure_logfire(settings: Settings | None = None) -> Any | None:
    """Configure and return the `logfire` module, or None if not configured."""
    settings = settings or get_settings()

    if not settings.logfire_token:
        return None

    try:
        import logfire
    except ImportError:
        logger.warning("LOGFIRE_TOKEN is set but the 'logfire' package is not installed")
        return None

    logfire.configure(token=settings.logfire_token, service_name=settings.qm_app_name)
    return logfire
