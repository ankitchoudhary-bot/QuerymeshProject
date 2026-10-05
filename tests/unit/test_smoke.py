"""Phase 0 smoke test.

Verifies:
  - the package imports
  - Settings loads with sane defaults
  - the health check passes end to end
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    import querymesh  # noqa: F401
    import querymesh.agents  # noqa: F401
    import querymesh.graph  # noqa: F401
    import querymesh.retrieval  # noqa: F401
    import querymesh.ingestion  # noqa: F401
    import querymesh.indexing  # noqa: F401
    import querymesh.embeddings  # noqa: F401
    import querymesh.mcp  # noqa: F401
    import querymesh.a2a  # noqa: F401
    import querymesh.llm  # noqa: F401
    import querymesh.cache  # noqa: F401
    import querymesh.observability  # noqa: F401
    import querymesh.evaluation  # noqa: F401
    import querymesh.api  # noqa: F401
    import querymesh.ui  # noqa: F401
    import querymesh.config  # noqa: F401


@pytest.mark.unit
def test_settings_defaults() -> None:
    from querymesh.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.qm_env in {"development", "test", "production"}
    assert settings.qm_app_name == "querymesh"
    assert settings.arxiv_max_results == 20
    assert settings.a2a_max_retries == 2


@pytest.mark.unit
def test_health_check_passes() -> None:
    from querymesh.health import run_health_check

    assert run_health_check() is True


@pytest.mark.unit
def test_logger_returns_logger_instance() -> None:
    import logging

    from querymesh.observability.logging import get_logger

    logger = get_logger("querymesh.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "querymesh.test"
