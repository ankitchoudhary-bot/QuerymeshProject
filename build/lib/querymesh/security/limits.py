"""Production safety limits: maximum query length, maximum document
count, and the validation helpers that enforce them.

Rule: "input validation, output validation... maximum query length,
maximum document count, maximum graph recursion." Graph recursion is
enforced separately at the LangGraph invocation layer (see
`querymesh.graph.workflow`'s `recursion_limit`); this module covers the
two limits that are about data shape, not control flow.
"""

from __future__ import annotations

from querymesh.errors import ValidationError

MAX_QUERY_LENGTH = 2000
MAX_TOP_K = 100
MIN_TOP_K = 1


def validate_query(query: str) -> str:
    """Validate a user-supplied research question.

    Returns the (stripped) query on success. Raises `ValidationError` for
    empty or excessively long input — the two shapes of bad input that
    could otherwise reach an LLM prompt or a retrieval query unchecked.
    """
    if query is None:
        raise ValidationError("query must not be None")

    stripped = query.strip()
    if not stripped:
        raise ValidationError("query must not be empty")

    if len(stripped) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"query length {len(stripped)} exceeds maximum of {MAX_QUERY_LENGTH} characters"
        )

    return stripped


def validate_top_k(top_k: int) -> int:
    """Validate a requested document count. Raises `ValidationError` for
    out-of-range values rather than silently clamping — silent clamping
    can mask a caller bug (e.g. an accidentally huge top_k) instead of
    surfacing it.
    """
    if top_k < MIN_TOP_K:
        raise ValidationError(f"top_k must be at least {MIN_TOP_K}, got {top_k}")
    if top_k > MAX_TOP_K:
        raise ValidationError(f"top_k of {top_k} exceeds maximum of {MAX_TOP_K}")
    return top_k
