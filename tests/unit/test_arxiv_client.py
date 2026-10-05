"""Phase 2 tests: ArxivClient search, pagination, retries, dedup — all via
httpx.MockTransport so no real network call is ever made.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from querymesh.config import Settings
from querymesh.ingestion import ArxivClient, ArxivError, PaperMetadata
from querymesh.ingestion.atom_parser import parse_feed
from tests.unit.fixtures_arxiv import EMPTY_FEED, make_entry, make_feed


def _settings(rate_limit: float = 0.0) -> Settings:
    return Settings(ARXIV_RATE_LIMIT_SECONDS=rate_limit, ARXIV_MAX_RESULTS=20)


def _client_with_handler(handler, rate_limit: float = 0.0) -> ArxivClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    return ArxivClient(settings=_settings(rate_limit), http_client=http_client)


@pytest.mark.unit
def test_parse_feed_extracts_all_fields() -> None:
    xml_text = make_feed([make_entry("2608.13558v1", title="Attention Evolutions")])
    papers = parse_feed(xml_text)

    assert len(papers) == 1
    paper = papers[0]
    assert isinstance(paper, PaperMetadata)
    assert paper.arxiv_id == "2608.13558v1"
    assert paper.title == "Attention Evolutions"
    assert paper.authors == ["Jane Doe"]
    assert "abstract" in paper.abstract.lower()
    assert "cs.CL" in paper.categories
    assert paper.pdf_url == "http://arxiv.org/pdf/2608.13558v1"
    assert paper.doi == "10.1000/2608.13558v1"
    assert paper.published is not None


@pytest.mark.unit
def test_parse_feed_empty_returns_empty_list() -> None:
    assert parse_feed(EMPTY_FEED) == []


@pytest.mark.unit
async def test_search_returns_parsed_papers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        xml_text = make_feed([make_entry("2608.00001v1"), make_entry("2608.00002v1")])
        return httpx.Response(200, text=xml_text)

    client = _client_with_handler(handler)
    try:
        papers = await client.search("transformer attention", max_results=10)
    finally:
        await client.aclose()

    assert len(papers) == 2
    assert {p.arxiv_id for p in papers} == {"2608.00001v1", "2608.00002v1"}


@pytest.mark.unit
async def test_search_paginates_across_multiple_pages() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = parse_qs(urlparse(str(request.url)).query)
        start = int(params["start"][0])
        calls.append(start)

        if start == 0:
            entries = [make_entry(f"2608.{i:05d}v1") for i in range(100)]
        elif start == 100:
            entries = [make_entry(f"2608.{i:05d}v1") for i in range(100, 150)]
        else:
            entries = []
        return httpx.Response(200, text=make_feed(entries))

    client = _client_with_handler(handler)
    try:
        # max_results=150 exceeds the 100-per-page cap, forcing a second request.
        papers = await client.search("hybrid retrieval", max_results=150)
    finally:
        await client.aclose()

    assert len(papers) == 150
    assert calls == [0, 100]


@pytest.mark.unit
async def test_search_deduplicates_across_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=make_feed([make_entry("2608.99999v1")]))

    client = _client_with_handler(handler)
    try:
        first = await client.search("query one", max_results=1)
        second = await client.search("query one", max_results=1)
    finally:
        await client.aclose()

    assert len(first) == 1
    assert len(second) == 0  # already seen by this client instance


@pytest.mark.unit
async def test_search_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            return httpx.Response(500, text="internal error")
        return httpx.Response(200, text=make_feed([make_entry("2608.11111v1")]))

    client = _client_with_handler(handler)
    try:
        papers = await client.search("retry test", max_results=1)
    finally:
        await client.aclose()

    assert attempts["count"] == 3
    assert len(papers) == 1


@pytest.mark.unit
async def test_search_raises_arxiv_error_after_exhausting_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = ArxivClient(settings=_settings(), http_client=http_client, max_retries=1)

    with pytest.raises(ArxivError):
        try:
            await client.search("always fails", max_results=1)
        finally:
            await client.aclose()


@pytest.mark.unit
async def test_search_zero_max_results_returns_empty_without_request() -> None:
    called = {"yes": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["yes"] = True
        return httpx.Response(200, text=EMPTY_FEED)

    client = _client_with_handler(handler)
    try:
        papers = await client.search("no results wanted", max_results=0)
    finally:
        await client.aclose()

    assert papers == []
    assert called["yes"] is False


@pytest.mark.unit
async def test_async_context_manager_closes_owned_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=make_feed([make_entry("2608.22222v1")]))

    transport = httpx.MockTransport(handler)

    async with ArxivClient(settings=_settings(), transport=transport) as client:
        papers = await client.search("context manager test", max_results=1)
        assert len(papers) == 1
        owned_client = client._client

    assert owned_client is not None
    assert owned_client.is_closed
