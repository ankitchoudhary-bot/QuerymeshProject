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

from querymesh.ingestion.document_models import ParsedDocument, Section, Table, Figure
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
    
    Extracts:
            - sections
            - tables
            - table markdown
            - table images
            - figures/images
            - figure captions
            - references
    """

    def __init__(self, image_dir: Path = Path("data/images"), images_scale: float = 2.0, ) -> None:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
        except ImportError as exc:
            raise ParsingError(
                "The 'docling' package is required for DoclingParser. "
                "Install it with `pip install docling`."
            ) from exc
        self.image_dir = Path(image_dir)
        self.images_scale = images_scale

    def parse(self, pdf_path: Path, arxiv_id: str, title: str, abstract: str) -> ParsedDocument:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import (DocumentConverter, PdfFormatOption, )


        if not pdf_path.exists():
            raise ParsingError(f"PDF not found at {pdf_path}")

        try:
            pipeline_options = PdfPipelineOptions()
            
            # Table structure extraction.
            pipeline_options.do_table_structure = True

            # Generate cropped images for tables and figures.
            pipeline_options.generate_table_images = True
            pipeline_options.generate_picture_images = True

            # Higher-resolution crops.
            pipeline_options.images_scale = self.images_scale

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
            
            result = converter.convert(str(pdf_path))
            doc = result.document
                        
        except Exception as exc:  # noqa: BLE001 - docling can raise many error types
            raise ParsingError(f"Docling failed to parse {pdf_path}: {exc}") from exc
        
        
        # Directory for this paper's extracted assets.
        paper_image_dir = (self.image_dir / arxiv_id)

        sections = _extract_sections(doc)

        tables = _extract_tables(doc=doc, image_dir=paper_image_dir, arxiv_id=arxiv_id, )

        figures = _extract_figures(doc=doc, image_dir=paper_image_dir, arxiv_id=arxiv_id, )
        
        references = _extract_references(doc)
        
        num_pages = getattr(doc, "num_pages", None, )

        if callable(num_pages):
            try:
                num_pages = num_pages()
            except Exception as exc:
                logger.debug("Could not determine page count: %s", exc,)
                num_pages = None

        logger.info(
            "Docling extracted %s sections, %s tables, "
            "%s figures from %s",
            len(sections),
            len(tables),
            len(figures),
            arxiv_id,
        )
        
        return ParsedDocument(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            sections=sections,
            tables=tables,
            figures=figures,
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


def _extract_tables(doc: object, image_dir: Path, arxiv_id: str, ) -> list[Table]:
    """Extract structured tables and their cropped images."""
    tables: list[Table] = []
    doc_tables = getattr(doc, "tables", None, ) or []
    
    for index, tbl in enumerate(doc_tables, start=1, ):
        rows: list[list[str]] = []
        markdown: str | None = None

        # -------------------------
        # Structured table data
        # -------------------------

        try:
            dataframe = tbl.export_to_dataframe(doc=doc)

            rows = dataframe.values.tolist()

            rows = [[str(cell) for cell in row] for row in rows]

        except Exception as exc:
            logger.warning("Could not extract rows for table %s: %s", index, exc, )

        # -------------------------
        # Markdown representation
        # -------------------------

        try:
            markdown = tbl.export_to_markdown(doc=doc)

        except Exception as exc:
            logger.debug("Could not export table %s as markdown: %s", index, exc, )

        # -------------------------
        # Table image
        # -------------------------

        image_path: str | None = None

        try:
            image = tbl.get_image(doc)

            if image is not None:
                image_dir.mkdir(parents=True, exist_ok=True, )

                path = (image_dir / f"{arxiv_id}_table_{index}.png")

                image.save(path)

                image_path = str(path)

        except Exception as exc:
            logger.warning("Could not extract image for table %s: %s", index, exc, )

        # -------------------------
        # Page
        # -------------------------

        page = _get_page_number(tbl)

        # -------------------------
        # Caption
        # -------------------------
        
        caption = _get_caption(doc, tbl)

        # caption = getattr(tbl, "caption", None, )

        # if caption is not None:
        #     try:
        #         caption = str(caption)
        #     except Exception:
        #         caption = None

        tables.append(
            Table(
                caption=caption,
                page=page,
                rows=rows,
                markdown=markdown,
                image_path=image_path,
            )
        )

    return tables

def _extract_figures(doc: object, image_dir: Path, arxiv_id: str, ) -> list[Figure]:
    """Extract figures/pictures and their cropped images."""

    figures: list[Figure] = []

    pictures = getattr(doc, "pictures", None, ) or []

    for index, picture in enumerate(pictures, start=1, ):
        image_path: str | None = None

        # -------------------------
        # Figure image
        # -------------------------
        try:
            image = picture.get_image(doc)

            if image is None:
                logger.debug("Picture %s has no image", index, )
                continue

            image_dir.mkdir(parents=True, exist_ok=True,)
            path = (image_dir / f"{arxiv_id}_figure_{index}.png")
            image.save(path)
            image_path = str(path)

        except Exception as exc:
            logger.warning("Could not extract figure %s: %s", index, exc, )

        # caption = getattr(picture, "caption", None, )

        # if caption is not None:
        #     try:
        #         caption = str(caption)
        #     except Exception:
        #         caption = None
        
        # -------------------------
        # Caption
        # -------------------------
        caption = _get_caption(doc, picture)
        
        # -------------------------
        # Page
        # -------------------------
        page = _get_page_number(picture)

        figures.append(
            Figure(
                caption=caption,
                page=page,
                image_path=image_path,
            )
        )

    return figures

def _get_page_number(item: object) -> int | None:
    """Best-effort extraction of the item's page number."""

    try:
        provenance = getattr(item, "prov", None, )

        if not provenance:
            return None

        first = provenance[0]

        page_no = getattr(first, "page_no", None, )

        if page_no is not None:
            return int(page_no)

    except Exception as exc:
        logger.debug("Could not determine item page: %s", exc, )

    return None

def _get_caption(doc: object, item: object) -> str | None:
    """Extract the caption text associated with a Docling item."""

    try:
        caption_text = getattr(item, "caption_text", None)

        if callable(caption_text):
            caption_text = caption_text(doc)

        if caption_text:
            text = str(caption_text).strip()
            if text:
                return text

    except Exception as exc:
        logger.debug("Could not extract caption using caption_text(): %s", exc, )

    return None

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
        