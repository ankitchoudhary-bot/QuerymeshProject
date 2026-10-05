"""Graph-wide constants.

Centralized so `MAX_RETRIES` has exactly one definition instead of being
duplicated across `state.py`, `workflow.py`, etc.
"""

from __future__ import annotations

MAX_RETRIES = 2
