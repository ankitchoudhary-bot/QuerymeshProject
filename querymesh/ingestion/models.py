"""Data models for the ingestion layer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Metadata for a single arXiv paper, as returned by search/fetch."""

    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str
    categories: list[str] = Field(default_factory=list)
    published: datetime | None = None
    updated: datetime | None = None
    pdf_url: str | None = None
    source_url: str | None = None
    doi: str | None = None

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.arxiv_id}: {self.title}"
