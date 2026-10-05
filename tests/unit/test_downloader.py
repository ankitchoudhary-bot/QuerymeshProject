"""Phase 3 tests: PdfDownloader — deterministic storage, dedup, retries,
validation. All via httpx.MockTransport, no live network or disk outside
tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from querymesh.ingestion.downloader import PdfDownloadError, PdfDownloader

_VALID_PDF_BYTES = b"%PDF-1.4\n%fake pdf content\n%%EOF"


@pytest.mark.unit
async def test_download_writes_file_and_returns_path(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_VALID_PDF_BYTES)

    downloader = PdfDownloader(storage_dir=tmp_path, transport=httpx.MockTransport(handler))
    try:
        path = await downloader.download("2608.00001v1", "http://arxiv.org/pdf/2608.00001v1")
    finally:
        await downloader.aclose()

    assert path == tmp_path / "2608.00001v1.pdf"
    assert path.exists()
    assert path.read_bytes() == _VALID_PDF_BYTES


@pytest.mark.unit
async def test_download_skips_existing_file(tmp_path: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, content=_VALID_PDF_BYTES)

    downloader = PdfDownloader(storage_dir=tmp_path, transport=httpx.MockTransport(handler))
    try:
        await downloader.download("2608.00002v1", "http://arxiv.org/pdf/2608.00002v1")
        await downloader.download("2608.00002v1", "http://arxiv.org/pdf/2608.00002v1")
    finally:
        await downloader.aclose()

    assert calls["count"] == 1  # second call was a no-op due to existing file


@pytest.mark.unit
async def test_download_force_redownloads(tmp_path: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, content=_VALID_PDF_BYTES)

    downloader = PdfDownloader(storage_dir=tmp_path, transport=httpx.MockTransport(handler))
    try:
        await downloader.download("2608.00003v1", "http://arxiv.org/pdf/2608.00003v1")
        await downloader.download("2608.00003v1", "http://arxiv.org/pdf/2608.00003v1", force=True)
    finally:
        await downloader.aclose()

    assert calls["count"] == 2


@pytest.mark.unit
async def test_download_rejects_non_pdf_content(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not a pdf</html>")

    downloader = PdfDownloader(
        storage_dir=tmp_path, transport=httpx.MockTransport(handler), max_retries=0
    )
    try:
        with pytest.raises(PdfDownloadError):
            await downloader.download("2608.00004v1", "http://arxiv.org/pdf/2608.00004v1")
    finally:
        await downloader.aclose()

    assert not downloader.path_for("2608.00004v1").exists()


@pytest.mark.unit
async def test_download_retries_then_succeeds(tmp_path: Path) -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(500)
        return httpx.Response(200, content=_VALID_PDF_BYTES)

    downloader = PdfDownloader(storage_dir=tmp_path, transport=httpx.MockTransport(handler))
    try:
        path = await downloader.download("2608.00005v1", "http://arxiv.org/pdf/2608.00005v1")
    finally:
        await downloader.aclose()

    assert attempts["count"] == 3
    assert path.exists()


@pytest.mark.unit
async def test_download_raises_after_exhausting_retries(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    downloader = PdfDownloader(
        storage_dir=tmp_path, transport=httpx.MockTransport(handler), max_retries=1
    )
    try:
        with pytest.raises(PdfDownloadError):
            await downloader.download("2608.00006v1", "http://arxiv.org/pdf/2608.00006v1")
    finally:
        await downloader.aclose()


@pytest.mark.unit
def test_exists_reflects_disk_state(tmp_path: Path) -> None:
    downloader = PdfDownloader(storage_dir=tmp_path)
    assert downloader.exists("2608.00007v1") is False

    downloader.path_for("2608.00007v1").write_bytes(_VALID_PDF_BYTES)
    assert downloader.exists("2608.00007v1") is True
