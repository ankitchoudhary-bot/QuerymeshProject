"""Research-paper-aware chunking.

Rule: "Do not blindly split every N characters." This chunker respects
paper structure first:

  1. The abstract becomes its own pseudo-section ("Abstract").
  2. Within each section, paragraphs are packed into a chunk up to
     `max_chars`, never splitting a paragraph unless the paragraph alone
     exceeds `max_chars` — in which case (and only then) it's split on
     sentence boundaries as a fallback.
  3. A chunk never mixes paragraphs from two different sections.

Chunk IDs are assigned in document order and are deterministic given the
same `ParsedDocument`, so re-chunking is idempotent.
"""

from __future__ import annotations

import re

from querymesh.ingestion.chunk_models import PaperChunk, make_chunk_id
from querymesh.ingestion.document_models import ParsedDocument, Section, Table, Figure

_DEFAULT_MAX_CHARS = 1200

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class PaperChunker:
    """Chunks a `ParsedDocument` into structure-aware `PaperChunk`s.
        Produces three types of chunks:

        text   -> normal section/abstract content
        table  -> structured table content
        figure -> figure caption/description
    """

    def __init__(self, max_chars: int = _DEFAULT_MAX_CHARS) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        self.max_chars = max_chars

    def chunk(self, doc: ParsedDocument) -> list[PaperChunk]:
        chunks: list[PaperChunk] = []
        index = 0
        
        # --------------------------------------------------
        # 1. Normal text chunks
        # --------------------------------------------------
        
        virtual_sections: list[Section] = []
        if doc.abstract.strip():
            virtual_sections.append(Section(title="Abstract", level=0, paragraphs=[doc.abstract]))
        virtual_sections.extend(doc.sections)

        for section in virtual_sections:
            for text in self._pack_section(section):
                chunks.append(
                    PaperChunk(
                        chunk_id=make_chunk_id(doc.arxiv_id, index, "text"),
                        arxiv_id=doc.arxiv_id,
                        title=doc.title,
                        section=section.title,
                        text=text,
                        page=section.page_start,
                        chunk_index=index,
                        chunk_type='text',
                        categories=doc.categories,
                    )
                )
                index += 1
                
        # --------------------------------------------------
        # 2. Table chunks
        # --------------------------------------------------

        for table in doc.tables:
            text = self._table_to_text(table)

            if not text.strip():
                continue

            chunks.append(
                PaperChunk(
                    chunk_id=make_chunk_id(doc.arxiv_id, index, "table"),
                    arxiv_id=doc.arxiv_id,
                    title=doc.title,
                    section=table.section,
                    text=text,
                    page=table.page,
                    chunk_index=index,
                    chunk_type="table",
                    image_path=table.image_path,
                    categories=doc.categories,
                )
            )

            index += 1
            
        # --------------------------------------------------
        # 3. Figure chunks
        # --------------------------------------------------

        for figure in doc.figures:

            text = self._figure_to_text(figure)

            if not text.strip():
                continue

            chunks.append(
                PaperChunk(
                    chunk_id=make_chunk_id(doc.arxiv_id, index, "figure"),
                    arxiv_id=doc.arxiv_id,
                    title=doc.title,
                    section=figure.section,
                    text=text,
                    page=figure.page,
                    chunk_index=index,
                    chunk_type="figure",
                    image_path=figure.image_path,
                    categories=doc.categories,
                )
            )

            index += 1

        return chunks
    
    # ------------------------------------------------------
    # Text
    # ------------------------------------------------------

    def _pack_section(self, section: Section) -> list[str]:
        """Pack a section's paragraphs into <=max_chars text blocks."""
        blocks: list[str] = []
        buffer: list[str] = []
        buffer_len = 0

        def flush() -> None:
            nonlocal buffer, buffer_len
            if buffer:
                blocks.append("\n\n".join(buffer))
                buffer = []
                buffer_len = 0

        for paragraph in section.paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) > self.max_chars:
                # Paragraph alone exceeds the limit: flush what we have,
                # then split this paragraph on sentence boundaries.
                flush()
                blocks.extend(_split_long_paragraph(paragraph, self.max_chars))
                continue

            # Would adding this paragraph overflow the current buffer?
            projected = buffer_len + (2 if buffer else 0) + len(paragraph)
            if projected > self.max_chars and buffer:
                flush()

            buffer.append(paragraph)
            buffer_len += (2 if len(buffer) > 1 else 0) + len(paragraph)

        flush()
        return blocks
    
    # ------------------------------------------------------
    # Tables
    # ------------------------------------------------------

    @staticmethod
    def _table_to_text(table: Table, ) -> str:

        parts: list[str] = []
        parts.append("[TABLE]")

        if table.caption:
            parts.append(f"Caption: {table.caption}")

        if table.markdown:
            parts.append(table.markdown.strip())

        elif table.rows:
            for row in table.rows:
                parts.append(" | ".join(row))

        return "\n\n".join(part for part in parts if part)
    
    # ------------------------------------------------------
    # Figures
    # ------------------------------------------------------
    @staticmethod
    def _figure_to_text(figure: Figure, ) -> str:

        parts: list[str] = []
        parts.append("[FIGURE]")

        if figure.caption:
            parts.append(f"Caption: {figure.caption}")
        
        if figure.description:
            parts.append(f"Description: {figure.description}")

        return "\n\n".join(part for part in parts if part)


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries, packing
    sentences up to max_chars each. Falls back to a hard character cut
    for a single sentence that is itself longer than max_chars.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s]
    pieces: list[str] = []
    buffer = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            # Hard cut as a last resort.
            for i in range(0, len(sentence), max_chars):
                pieces.append(sentence[i : i + max_chars])
            continue

        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if len(candidate) > max_chars:
            if buffer:
                pieces.append(buffer)
            buffer = sentence
        else:
            buffer = candidate

    if buffer:
        pieces.append(buffer)

    return pieces