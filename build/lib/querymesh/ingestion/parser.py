"""PDF parsing abstraction.

`DocumentParser` is the interface the rest of the app depends on.
`DoclingParser` is the real implementation, but it imports the `docling`
library lazily (inside `parse()`) so that importing this module — or any
module that imports it — never requires `docling` and its heavy ML
dependencies (torch, transformers, etc.) to be installed. Install it with:

    pip install docling

`FakeParser` is a first-class, dependency-free implementation used for
tests and for developing the rest of the pipeline (chunking, indexing)
before Docling is available in a given environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from querymesh.ingestion.document_models import ParsedDocument, Section, Table
from querymesh.ingestion.errors import IngestionError
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


class ParsingError(IngestionError):
    """Raised when a PDF cannot be parsed into a structured document."""


@runtime_checkable
class DocumentParser(Protocol):
    def parse(self, pdf_path: Path, arxiv_id: str, title: str, abstract: str) -> ParsedDocument:
        ...


class DoclingParser:
    """Parses PDFs into `ParsedDocument` using the Docling library.

    Title and abstract are passed in (rather than re-derived from the PDF)
    because the arXiv metadata already has them reliably — Docling is used
    here for what it's good at: sections, tables, and reference extraction
    from the PDF body.
    """

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
        except ImportError as exc:
            raise ParsingError(
                "The 'docling' package is required for DoclingParser. "
                "Install it with `pip install docling`."
            ) from exc

    def parse(self, pdf_path: Path, arxiv_id: str, title: str, abstract: str) -> ParsedDocument:
        from docling.document_converter import DocumentConverter

        if not pdf_path.exists():
            raise ParsingError(f"PDF not found at {pdf_path}")

        try:
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            doc = result.document
        except Exception as exc:  # noqa: BLE001 - docling can raise many error types
            raise ParsingError(f"Docling failed to parse {pdf_path}: {exc}") from exc

        sections = _extract_sections(doc)
        tables = _extract_tables(doc)
        references = _extract_references(doc)
        num_pages = getattr(doc, "num_pages", None)
        if callable(num_pages):
            try:
                num_pages = num_pages()
            except Exception as exc:  # noqa: BLE001 - degrade gracefully, but log it
                logger.debug("Could not determine page count: %s", exc)
                num_pages = None

        return ParsedDocument(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            sections=sections,
            tables=tables,
            references=references,
            num_pages=num_pages,
        )


def _extract_sections(doc: object) -> list[Section]:
    """Best-effort section extraction from a Docling document object.

    Docling's public API for walking headings/paragraphs has moved across
    versions; this stays defensive (getattr-based) rather than assuming one
    exact shape, and returns an empty list rather than raising if the
    expected structure isn't present.
    """
    sections: list[Section] = []
    texts = getattr(doc, "texts", None) or []

    current_title = "Body"
    current_paragraphs: list[str] = []

    def flush() -> None:
        if current_paragraphs:
            sections.append(Section(title=current_title, paragraphs=list(current_paragraphs)))

    for item in texts:
        label = str(getattr(item, "label", "")).lower()
        text = getattr(item, "text", "") or ""
        if not text.strip():
            continue
        if "heading" in label or "title" in label and item is not texts[0]:
            flush()
            current_title = text.strip()
            current_paragraphs = []
        else:
            current_paragraphs.append(text.strip())

    flush()
    return sections


def _extract_tables(doc: object) -> list[Table]:
    tables: list[Table] = []
    for tbl in getattr(doc, "tables", None) or []:
        try:
            rows = tbl.export_to_dataframe().values.tolist()
            rows = [[str(cell) for cell in row] for row in rows]
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, but log it
            logger.debug("Could not extract table rows: %s", exc)
            rows = []
        tables.append(Table(caption=getattr(tbl, "caption", None), rows=rows))
    return tables


def _extract_references(doc: object) -> list[str]:
    # Docling doesn't have a dedicated "references" concept in older
    # versions; approximate by looking for a section literally titled
    # "References" among extracted sections.
    references: list[str] = []
    for section in _extract_sections(doc):
        if section.title.strip().lower() in {"references", "bibliography"}:
            references.extend(section.paragraphs)
    return references


class FakeParser:
    """Dependency-free parser used for tests and offline development."""

    def parse(self, pdf_path: Path, arxiv_id: str, title: str, abstract: str) -> ParsedDocument:
        return ParsedDocument(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            sections=[
                Section(title="Introduction", level=1, paragraphs=["This is a fake introduction."]),
                Section(title="Conclusion", level=1, paragraphs=["This is a fake conclusion."]),
            ],
            tables=[],
            references=["Fake Author. Fake Paper Title. 2026."],
            num_pages=1,
        )
