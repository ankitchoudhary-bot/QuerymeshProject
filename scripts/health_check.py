#!/usr/bin/env python
"""Convenience script: run the QueryMesh health check directly.

Equivalent to `python -m querymesh health`, provided here because the
target repository structure keeps operational scripts under `scripts/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly (`python scripts/health_check.py`)
# without having installed the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from querymesh.health import run_health_check  # noqa: E402


if __name__ == "__main__":
    ok = run_health_check()
    raise SystemExit(0 if ok else 1)
