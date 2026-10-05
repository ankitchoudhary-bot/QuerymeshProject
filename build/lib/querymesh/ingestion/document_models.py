"""Structured document models produced by PDF parsing.

These preserve paper structure (sections, subsections, tables, references)
rather than collapsing everything into a single blob of text, so that
Phase 4 chunking can be structure-aware instead of splitting blindly.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Table(BaseModel):
    """A table extracted from the document."""

    caption: str | None = None
    section: str | None = None
    page: int | None = None
    # Row-major cell text; kept as raw strings rather than typed values
    # since papers' tables vary too much to type generically.
    rows: list[list[str]] = Field(default_factory=list)


class Section(BaseModel):
    """A section (or subsection) of a paper."""

    title: str
    level: int = 1  # 1 = top-level section, 2 = subsection, etc.
    paragraphs: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class ParsedDocument(BaseModel):
    """The full structured output of parsing one paper's PDF."""

    arxiv_id: str
    title: str
    abstract: str
    sections: list[Section] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    num_pages: int | None = None

    @property
    def full_text(self) -> str:
        """Flat text view — abstract + all section paragraphs, in order.

        Useful as a fallback for anything not yet structure-aware; the
        chunker (Phase 4) should prefer `sections` over this.
        """
        parts = [self.abstract]
        for section in self.sections:
            parts.append(section.title)
            parts.extend(section.paragraphs)
        return "\n\n".join(p for p in parts if p)
