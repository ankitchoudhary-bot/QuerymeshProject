"""Shared helper for parsing structured (JSON) output from an `LLM`.

Real LLMs asked to "respond only in JSON" occasionally wrap the JSON in
prose or code fences anyway — this strips common wrappers before parsing,
and raises a clear error rather than silently returning `None` so callers
can decide their own fallback behavior.
"""

from __future__ import annotations

import json
import re

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class StructuredOutputError(Exception):
    """Raised when an LLM response can't be parsed as the expected JSON shape."""


def parse_json_response(text: str) -> dict | list:
    cleaned = _CODE_FENCE_RE.sub("", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Could not parse LLM response as JSON: {exc}") from exc
