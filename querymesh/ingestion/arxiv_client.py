"""Production-oriented arXiv API client.

Handles search with pagination, retries with backoff, basic client-side
rate limiting (arXiv asks for a few seconds between requests), and
deduplication of papers already seen by this client instance.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from querymesh.config import Settings, get_settings
from querymesh.ingestion.atom_parser import parse_feed
from querymesh.ingestion.errors import ArxivError
from querymesh.ingestion.models import PaperMetadata
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

_MAX_PAGE_SIZE = 100
_REQUEST_TIMEOUT_SECONDS = 30.0
_RETRY_BASE_DELAY_SECONDS = 0.5


class ArxivClient:
    """Async client for the arXiv search API.

    Pass a pre-built `http_client` (e.g. one wired to `httpx.MockTransport`)
    to test without hitting the network. When omitted, an `httpx.AsyncClient`
    is created and owned by this instance.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
    ) -> None:
        settings = settings or get_settings()
        self.base_url = settings.arxiv_base_url
        self.rate_limit_seconds = settings.arxiv_rate_limit_seconds
        self.default_max_results = settings.arxiv_max_results
        self.max_retries = max_retries

        self._transport = transport
        self._client = http_client
        self._owns_client = http_client is None
        self._last_request_monotonic: float | None = None
        self._seen_ids: set[str] = set()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT_SECONDS, transport=self._transport
            )
        return self._client

    async def _respect_rate_limit(self) -> None:
        if self._last_request_monotonic is not None:
            elapsed = time.monotonic() - self._last_request_monotonic
            remaining = self.rate_limit_seconds - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
        self._last_request_monotonic = time.monotonic()

    async def _fetch_page(self, params: dict) -> str:
        client = await self._get_client()
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            await self._respect_rate_limit()
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "arXiv request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2**attempt))

        raise ArxivError(
            f"arXiv request failed after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    async def search(
        self,
        query: str,
        max_results: int | None = None,
        start: int = 0,
    ) -> list[PaperMetadata]:
        """Search arXiv, transparently paginating until `max_results` is met.

        Papers already returned by a previous `search()` call on this
        client instance are skipped (deduplication across calls).
        """
        max_results = max_results if max_results is not None else self.default_max_results
        if max_results <= 0:
            return []

        results: list[PaperMetadata] = []
        current_start = start
        fetched_total = 0

        while fetched_total < max_results:
            batch_size = min(_MAX_PAGE_SIZE, max_results - fetched_total)
            params = {
                "search_query": query,
                "start": current_start,
                "max_results": batch_size,
            }
            xml_text = await self._fetch_page(params)
            batch = parse_feed(xml_text)

            if not batch:
                break

            for paper in batch:
                fetched_total += 1
                if paper.arxiv_id in self._seen_ids:
                    continue
                self._seen_ids.add(paper.arxiv_id)
                results.append(paper)

            current_start += len(batch)

            if len(batch) < batch_size:
                # arXiv returned fewer results than asked for: no more pages.
                break

        return results

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "ArxivClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
