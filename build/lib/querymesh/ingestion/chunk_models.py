"""Chunk model produced by the chunker.

Chunk IDs are deterministic: `<arxiv_id>:chunk:<0-padded index>`, e.g.
`2608.13558v1:chunk:0001`. Determinism matters because downstream indexing
(Phase 5) needs re-running the chunker on the same document to produce the
same IDs, so re-indexing is idempotent rather than creating duplicates.
"""

from __future__ import annotations

from pydantic import BaseModel


class PaperChunk(BaseModel):
    chunk_id: str
    arxiv_id: str
    title: str
    section: str | None
    text: str
    page: int | None
    chunk_index: int


def make_chunk_id(arxiv_id: str, chunk_index: int) -> str:
    return f"{arxiv_id}:chunk:{chunk_index:04d}"
