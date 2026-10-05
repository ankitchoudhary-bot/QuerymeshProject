"""Phase 3 tests: parser abstraction (FakeParser + DoclingParser import
guard) and the end-to-end IngestionPipeline (metadata -> download -> parse).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from querymesh.ingestion import (
    DoclingParser,
    FakeParser,
    IngestionPipeline,
    ParsedDocument,
    ParsingError,
    PdfDownloader,
)
from querymesh.ingestion.models import PaperMetadata

_VALID_PDF_BYTES = b"%PDF-1.4\n%fake pdf content\n%%EOF"


@pytest.mark.unit
def test_fake_parser_returns_structured_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "2608.00001v1.pdf"
    pdf_path.write_bytes(_VALID_PDF_BYTES)

    parser = FakeParser()
    doc = parser.parse(pdf_path, arxiv_id="2608.00001v1", title="Test Paper", abstract="An abstract.")

    assert isinstance(doc, ParsedDocument)
    assert doc.arxiv_id == "2608.00001v1"
    assert doc.title == "Test Paper"
    assert len(doc.sections) == 2
    assert doc.references
    assert "abstract" in doc.full_text.lower() or "An abstract." in doc.full_text


@pytest.mark.unit
def test_docling_parser_raises_parsing_error_when_not_installed() -> None:
    # In this environment docling is intentionally not installed (heavy ML
    # dependency) — verify the abstraction fails loudly and clearly rather
    # than with a raw ImportError deep in the pipeline.
    try:
        import docling  # noqa: F401

        pytest.skip("docling is installed in this environment; import-guard test not applicable")
    except ImportError:
        pass

    with pytest.raises(ParsingError, match="docling"):
        DoclingParser()


@pytest.mark.unit
async def test_pipeline_downloads_then_parses(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_VALID_PDF_BYTES)

    downloader = PdfDownloader(storage_dir=tmp_path, transport=httpx.MockTransport(handler))
    pipeline = IngestionPipeline(downloader=downloader, parser=FakeParser())

    paper = PaperMetadata(
        arxiv_id="2608.00042v1",
        title="Pipeline Test Paper",
        abstract="Pipeline test abstract.",
        pdf_url="http://arxiv.org/pdf/2608.00042v1",
    )

    try:
        doc = await pipeline.process(paper)
    finally:
        await downloader.aclose()

    assert doc.arxiv_id == "2608.00042v1"
    assert doc.title == "Pipeline Test Paper"
    assert (tmp_path / "2608.00042v1.pdf").exists()


@pytest.mark.unit
async def test_pipeline_rejects_paper_without_pdf_url(tmp_path: Path) -> None:
    downloader = PdfDownloader(storage_dir=tmp_path)
    pipeline = IngestionPipeline(downloader=downloader, parser=FakeParser())

    paper = PaperMetadata(arxiv_id="2608.00099v1", title="No PDF", abstract="...", pdf_url=None)

    with pytest.raises(ValueError):
        await pipeline.process(paper)


@pytest.mark.unit
async def test_pipeline_process_many_skips_failures(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_VALID_PDF_BYTES)

    downloader = PdfDownloader(storage_dir=tmp_path, transport=httpx.MockTransport(handler))
    pipeline = IngestionPipeline(downloader=downloader, parser=FakeParser())

    good_paper = PaperMetadata(
        arxiv_id="2608.00100v1",
        title="Good Paper",
        abstract="...",
        pdf_url="http://arxiv.org/pdf/2608.00100v1",
    )
    bad_paper = PaperMetadata(arxiv_id="2608.00101v1", title="Bad Paper", abstract="...", pdf_url=None)

    try:
        results = await pipeline.process_many([good_paper, bad_paper])
    finally:
        await downloader.aclose()

    assert len(results) == 1
    assert results[0].arxiv_id == "2608.00100v1"
