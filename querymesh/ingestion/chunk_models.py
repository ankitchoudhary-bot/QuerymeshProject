"""Chunk model produced by the chunker.

Chunk IDs are deterministic: `<arxiv_id>:chunk:<0-padded index>`, e.g.
`2608.13558v1:chunk:0001`. Determinism matters because downstream indexing
(Phase 5) needs re-running the chunker on the same document to produce the
same IDs, so re-indexing is idempotent rather than creating duplicates.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

ChunkType = Literal["text", "table", "figure"]

class PaperChunk(BaseModel):
    chunk_id: str
    arxiv_id: str
    title: str
    section: str | None
    text: str
    page: int | None
    chunk_index: int
    
    # Identifies what kind of content this chunk represents.
    chunk_type: ChunkType = "text"
    
    # Path to the original/cropped visual asset.
    # Used for tables and figures; None for normal text.
    image_path: str | None = None
    
    table_markdown: str | None = None
    
    # arXiv categories associated with the paper.
    categories: list[str] = Field(default_factory=list)


# def make_chunk_id(arxiv_id: str, chunk_index: int) -> str:
#     return f"{arxiv_id}:chunk:{chunk_index:04d}"

def make_chunk_id(arxiv_id: str, chunk_index: int, chunk_type: str = "text", ) -> str:
    return f"{arxiv_id}:{chunk_type}:{chunk_index:04d}"