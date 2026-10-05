"""Basic structured logging for QueryMesh.

This is intentionally minimal in Phase 0: stdlib `logging` configured with
a consistent formatter and level pulled from Settings. Phase 18
(Observability) will layer Logfire and Langfuse tracing on top of this
without replacing it — this remains the baseline logger used everywhere
in the codebase.

Rule reminder: never log secrets or API keys.
"""

from __future__ import annotations

import logging
import sys

from querymesh.config import get_settings

_CONFIGURED = False

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once per process.

    Safe to call multiple times — only the first call has an effect unless
    a different level is explicitly requested.
    """
    global _CONFIGURED

    settings = get_settings()
    resolved_level = (level or settings.qm_log_level).upper()

    if _CONFIGURED:
        logging.getLogger().setLevel(resolved_level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured."""
    configure_logging()
    return logging.getLogger(name)
