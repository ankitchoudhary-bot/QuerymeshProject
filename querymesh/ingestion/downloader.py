"""PDF acquisition for arXiv papers.

Downloads are stored deterministically at `<storage_dir>/<arxiv_id>.pdf` so
repeat runs (e.g. the Airflow DAG in Phase 20) never re-download a paper
that's already on disk, and can trivially detect what's missing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from querymesh.ingestion.errors import IngestionError
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

_PDF_MAGIC = b"%PDF-"
_REQUEST_TIMEOUT_SECONDS = 60.0
_RETRY_BASE_DELAY_SECONDS = 0.5


class PdfDownloadError(IngestionError):
    """Raised when a PDF cannot be downloaded or fails validation."""


class PdfDownloader:
    """Downloads and stores arXiv PDFs deterministically on disk."""

    def __init__(
        self,
        storage_dir: str | Path,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries

        self._transport = transport
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT_SECONDS, transport=self._transport
            )
        return self._client

    def path_for(self, arxiv_id: str) -> Path:
        """Deterministic on-disk path for a given arxiv_id."""
        safe_id = arxiv_id.replace("/", "_")
        return self.storage_dir / f"{safe_id}.pdf"

    def exists(self, arxiv_id: str) -> bool:
        return self.path_for(arxiv_id).exists()

    async def download(self, arxiv_id: str, pdf_url: str, force: bool = False) -> Path:
        """Download a paper's PDF, skipping the request entirely if it's
        already stored (unless `force=True`).
        """
        destination = self.path_for(arxiv_id)

        if destination.exists() and not force:
            logger.info("PDF for %s already downloaded, skipping", arxiv_id)
            return destination

        client = await self._get_client()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await client.get(pdf_url)
                response.raise_for_status()
                content = response.content
                _validate_pdf_bytes(content, arxiv_id)

                tmp_path = destination.with_suffix(".pdf.part")
                tmp_path.write_bytes(content)
                tmp_path.replace(destination)
                return destination
            except (httpx.HTTPError, PdfDownloadError) as exc:
                last_exc = exc
                logger.warning(
                    "PDF download failed for %s (attempt %d/%d): %s",
                    arxiv_id,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise PdfDownloadError(
            f"Failed to download PDF for {arxiv_id} after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "PdfDownloader":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _validate_pdf_bytes(content: bytes, arxiv_id: str) -> None:
    if not content:
        raise PdfDownloadError(f"Empty PDF content for {arxiv_id}")
    if not content.startswith(_PDF_MAGIC):
        raise PdfDownloadError(f"Downloaded content for {arxiv_id} is not a valid PDF")
