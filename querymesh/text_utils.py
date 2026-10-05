"""Small text-processing utilities shared across layers that don't
belong to any one of them (ingestion's chunker, agents' citation mapper).

Kept deliberately tiny and dependency-free — no NLP library — consistent
with the project's "don't over-engineer" rule. Good enough for chunk-size
fallback splitting (Phase 4) and sentence-to-chunk citation mapping
(Phase 11); not a substitute for real sentence-boundary detection
(abbreviations like "e.g." or "Dr." will over-split).
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on `.`/`!`/`?` boundaries.

    Strips whitespace and drops empty results; does not attempt to merge
    or fix up abbreviation-induced over-splitting.
    """
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]



import re


def _split_claim_units(answer: str) -> list[str]:
    """Split an answer into citation-worthy units while preserving list items."""

    units: list[str] = []

    for raw_line in answer.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        # Markdown headings are not factual claims by themselves.
        if line.startswith("#"):
            continue

        # Bullet item:
        # - something
        # * something
        if re.match(r"^[-*+]\s+", line):
            units.append(line)
            continue

        # Numbered item:
        # 1. something
        # 2. something
        if re.match(r"^\d+[.)]\s+", line):
            units.append(line)
            continue

        # Normal prose can still be sentence-split.
        units.extend(split_sentences(line))

    return units