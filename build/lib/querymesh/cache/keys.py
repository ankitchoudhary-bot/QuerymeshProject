"""Cache key construction.

query
 ↓
normalize
 ↓
SHA256
 ↓
cache key

Example: `querymesh:answer:<sha256>`. Normalization collapses whitespace
and case so trivially different phrasings of the same question ("What is
Attention?" vs " what is attention? ") hit the same cache entry.
"""

from __future__ import annotations

import hashlib

_DEFAULT_KEY_PREFIX = "querymesh:answer"


def normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def make_cache_key(query: str, prefix: str = _DEFAULT_KEY_PREFIX) -> str:
    normalized = normalize_query(query)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"
