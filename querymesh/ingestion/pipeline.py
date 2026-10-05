"""Ties together metadata, download, and parsing for one paper.

Phase 4 (chunking) will consume `ParsedDocument` from here. This module
deliberately does not decide *which* parser to use — callers inject a
`DocumentParser` (e.g. `FakeParser` for tests/dev, `DoclingParser` once
Docling is installed), keeping the pipeline dependency-injectable per the
project's architectural principles.
"""

from __future__ import annotations

from pathlib import Path

from querymesh.ingestion.document_models import ParsedDocument
from querymesh.ingestion.downloader import PdfDownloader
from querymesh.ingestion.models import PaperMetadata
from querymesh.ingestion.parser import DocumentParser
from querymesh.observability.logging import get_logger
from querymesh.ingestion.figure_enricher import FigureEnricher
from querymesh.vision.ollama import OllamaVisionModel
logger = get_logger(__name__)


class IngestionPipeline:
    """arXiv metadata -> PDF download -> Docling parse -> ParsedDocument."""

    def __init__(self, downloader: PdfDownloader, parser: DocumentParser) -> None:
        self.downloader = downloader
        self.parser = parser

        self.figure_enricher = FigureEnricher(vision_model=OllamaVisionModel(model="qwen3-vl:4b", base_url="http://172.28.240.1:11434", ))

    async def process(self, paper: PaperMetadata) -> ParsedDocument:
        if not paper.pdf_url:
            raise ValueError(f"Paper {paper.arxiv_id} has no pdf_url")

        pdf_path: Path = await self.downloader.download(paper.arxiv_id, paper.pdf_url)
        logger.info("Parsing %s from %s", paper.arxiv_id, pdf_path)

        parsed = self.parser.parse(
            pdf_path=pdf_path,
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
        )
        
        parsed= await self.figure_enricher.enrich(parsed)
        logger.info(
            "Finished processing %s: %s figures enriched",
            paper.arxiv_id,
            len(parsed.figures),
        )
        return parsed

    async def process_many(self, papers: list[PaperMetadata]) -> list[ParsedDocument]:
        results: list[ParsedDocument] = []
        for paper in papers:
            try:
                results.append(await self.process(paper))
            except Exception as exc:  # noqa: BLE001 - one bad paper shouldn't stop the batch
                logger.error("Failed to process %s: %s", paper.arxiv_id, exc)
        return results
