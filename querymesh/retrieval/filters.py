"""Shared metadata-filter clause builder for BM25/vector retrieval.

Filters are simple `{field: value}` or `{field: [values]}` dicts applied
as OpenSearch `term`/`terms` clauses against keyword fields (e.g.
`arxiv_id`, `categories`). Kept intentionally simple — range/date filters
can be added later if a real need shows up, per the "don't over-engineer"
rule.
"""

from __future__ import annotations


def build_filter_clauses(filters: dict[str, str | list[str]]) -> list[dict]:
    clauses: list[dict] = []
    for field, value in filters.items():
        if isinstance(value, list):
            clauses.append({"terms": {field: value}})
        else:
            clauses.append({"term": {field: value}})
    return clauses
