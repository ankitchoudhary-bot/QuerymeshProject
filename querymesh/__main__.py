"""QueryMesh CLI entrypoint.

Phase 0 only supports a health check. Later phases will extend this with
subcommands for ingestion, running the API, launching the Gradio UI, etc.

Usage:
    python -m querymesh health
"""

from __future__ import annotations

import sys

from querymesh.health import run_health_check
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    command = argv[0] if argv else "health"

    if command == "health":
        ok = run_health_check()
        return 0 if ok else 1

    logger.error("Unknown command: %s", command)
    print(f"Unknown command: {command!r}. Available commands: health")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
