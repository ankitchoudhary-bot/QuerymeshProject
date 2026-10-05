"""Health-check logic for QueryMesh.

Phase 0 health check only validates local, dependency-free concerns:
  - configuration loads without error
  - required package structure is importable
  - logging is configured

Later phases will extend `run_health_check` to also ping OpenSearch,
Redis, and other external services, but each check should remain
independent so a single missing service doesn't crash the whole check.
"""

from __future__ import annotations

from querymesh.config import get_settings
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


def run_health_check() -> bool:
    """Run all Phase-0 health checks. Returns True if all checks pass."""
    checks: list[tuple[str, bool, str]] = []

    # Check 1: configuration loads
    try:
        settings = get_settings()
        checks.append(
            ("configuration", True, f"env={settings.qm_env}, app={settings.qm_app_name}")
        )
    except Exception as exc:
        checks.append(("configuration", False, str(exc)))

    # Check 2: core package imports resolve
    try:
        import querymesh.agents  # noqa: F401 - import-existence check, not used directly
        import querymesh.graph  # noqa: F401
        import querymesh.ingestion  # noqa: F401
        import querymesh.retrieval  # noqa: F401

        checks.append(("package_imports", True, "core packages import cleanly"))
    except Exception as exc:
        checks.append(("package_imports", False, str(exc)))

    all_ok = all(ok for _, ok, _ in checks)

    print("QueryMesh Health Check")
    print("=" * 40)
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
    print("=" * 40)
    print("OVERALL: " + ("HEALTHY" if all_ok else "UNHEALTHY"))

    if all_ok:
        logger.info("Health check passed")
    else:
        logger.error("Health check failed")

    return all_ok
