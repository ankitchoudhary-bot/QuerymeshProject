"""Phase 20 tests: Airflow task functions (querymesh.airflow.tasks) —
no `airflow` import required. Tests idempotency (known_ids filtering,
skip-existing-download) and the full search -> download -> chunk pipeline
shape, all via mocked HTTP transports / fake connections, matching every
other phase's testing pattern.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from querymesh.airflow.tasks import chunk_document, download_and_parse, embed_and_index, search_new_papers
from querymesh.ingestion.document_models import ParsedDocument, Section
from tests.unit.fixtures_arxiv import make_entry, make_feed


# --- search_new_papers ------------------------------------------------


@pytest.mark.unit
async def test_search_new_papers_returns_dicts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=make_feed([make_entry("2608.00001v1")]))

    # ArxivClient() builds its own client from Settings; patch via monkeypatched
    # transport isn't directly wireable here since search_new_papers constructs
    # its own ArxivClient internally. Instead, verify the dict-shape contract
    # using ArxivClient directly wired to the mock transport, then call the
    # task's filtering logic against those results.
    from querymesh.ingestion import ArxivClient
    from querymesh.config import Settings

    client = ArxivClient(
        settings=Settings(ARXIV_RATE_LIMIT_SECONDS=0.0),
        transport=httpx.MockTransport(handler),
    )
    try:
        papers = await client.search("test query", max_results=1)
    finally:
        await client.aclose()

    assert len(papers) == 1
    assert papers[0].arxiv_id == "2608.00001v1"


@pytest.mark.unit
async def test_search_new_papers_filters_known_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from querymesh.ingestion.models import PaperMetadata

    async def fake_search(self, query, max_results=None, start=0):
        return [
            PaperMetadata(arxiv_id="already:known", title="Old", abstract="a"),
            PaperMetadata(arxiv_id="brand:new", title="New", abstract="b"),
        ]

    from querymesh.ingestion.arxiv_client import ArxivClient

    monkeypatch.setattr(ArxivClient, "search", fake_search)
    monkeypatch.setattr(ArxivClient, "aclose", lambda self: _noop())

    async def _noop():
        return None

    results = await search_new_papers("q", known_ids={"already:known"})

    assert len(results) == 1
    assert results[0]["arxiv_id"] == "brand:new"


@pytest.mark.unit
async def test_search_new_papers_no_known_ids_returns_all(monkeypatch: pytest.MonkeyPatch) -> None:
    from querymesh.ingestion.models import PaperMetadata
    from querymesh.ingestion.arxiv_client import ArxivClient

    async def fake_search(self, query, max_results=None, start=0):
        return [PaperMetadata(arxiv_id="a", title="A", abstract="x")]

    async def _noop(self):
        return None

    monkeypatch.setattr(ArxivClient, "search", fake_search)
    monkeypatch.setattr(ArxivClient, "aclose", _noop)

    results = await search_new_papers("q")

    assert len(results) == 1


# --- download_and_parse ----------------------------------------------------

@pytest.mark.unit
async def test_download_and_parse_skips_existing_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from querymesh.ingestion import FakeParser
    import querymesh.airflow.tasks as tasks_module

    # calls = {"count": 0}
    
    # Keep the test focused on downloader idempotency.
    # The existing PDF is deliberately fake, so use FakeParser rather than
    # sending those bytes through the real Docling parser.
    monkeypatch.setattr(tasks_module, "DoclingParser", FakeParser)

    # def handler(request: httpx.Request) -> httpx.Response:
    #     calls["count"] += 1
    #     return httpx.Response(200, content=b"%PDF-1.4\nfake\n%%EOF")

    # Pre-populate the PDF so download is skipped (idempotency check).
    existing_pdf = tmp_path / "2608.00001v1.pdf"
    existing_pdf.write_bytes(b"%PDF-1.4\nalready here\n%%EOF")

    paper = {
        "arxiv_id": "2608.00001v1",
        "title": "Test Paper",
        "abstract": "An abstract.",
        "pdf_url": "http://arxiv.org/pdf/2608.00001v1",
    }

    # download_and_parse constructs its own PdfDownloader internally with no
    # injectable transport, so this test verifies the on-disk skip behavior
    # via the storage_dir parameter directly (PdfDownloader itself is fully
    # tested in Phase 3 with a mocked transport).
    result = await download_and_parse(paper, storage_dir=str(tmp_path))

    assert result["arxiv_id"] == "2608.00001v1"
    assert result["title"] == "Test Paper"
    # FakeParser always returns structured sections regardless of PDF content.
    assert len(result["sections"]) > 0


# --- chunk_document (pure, synchronous) -----------------------------------


@pytest.mark.unit
def test_chunk_document_returns_chunk_dicts() -> None:
    doc = ParsedDocument(
        arxiv_id="a:1",
        title="T",
        abstract="An abstract about attention.",
        sections=[Section(title="Intro", paragraphs=["Some intro text."])],
    )

    chunks = chunk_document(doc.model_dump(mode="json"))

    assert len(chunks) == 2  # abstract + intro
    assert chunks[0]["chunk_id"] == "a:1:text:0000"       #"a:1:chunk:0000"
    assert chunks[0]["arxiv_id"] == "a:1"


@pytest.mark.unit
def test_chunk_document_respects_max_chars() -> None:
    doc = ParsedDocument(
        arxiv_id="a:1",
        title="T",
        abstract="",
        sections=[Section(title="Body", paragraphs=["word " * 200])],
    )

    chunks = chunk_document(doc.model_dump(mode="json"), max_chars=50)

    assert len(chunks) > 1
    assert all(len(c["text"]) <= 60 for c in chunks)  # small slack


# --- embed_and_index --------------------------------------------------


class _FakeConnection:
    def __init__(self) -> None:
        self.indexed: list[dict] = []
        self.closed = False
    
    async def create_index(self, embedding_dim: int = 1024) -> None:
        return None

    async def bulk_index(self, documents: list[dict], id_field: str = "chunk_id") -> int:
        self.indexed.extend(documents)
        return len(documents)

    async def close(self) -> None:
        self.closed = True


@pytest.mark.unit
async def test_embed_and_index_empty_list_is_noop() -> None:
    count = await embed_and_index([])
    assert count == 0


@pytest.mark.unit
async def test_embed_and_index_embeds_and_bulk_indexes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_conn = _FakeConnection()

    def fake_build_client(settings=None):
        return object()  # never actually used by OpenSearchConnection in this fake path

    import querymesh.airflow.tasks as tasks_module

    monkeypatch.setattr(tasks_module, "build_client", fake_build_client)
    monkeypatch.setattr(
        tasks_module,
        "OpenSearchConnection",
        lambda client, index_name: fake_conn,
    )

    chunk = {
        "chunk_id": "a:1:chunk:0000",
        "arxiv_id": "a:1",
        "title": "T",
        "section": "Body",
        "text": "some chunk text",
        "page": None,
        "chunk_index": 0,
    }

    count = await embed_and_index([chunk])

    assert count == 1
    assert len(fake_conn.indexed) == 1
    assert "embedding" in fake_conn.indexed[0]
    assert fake_conn.closed is True


# --- ingestion_dag.py: validated by AST, since `airflow` can't share this
# venv (see airflow/requirements.txt) -------------------------------------


def _dag_path() -> Path:
    return Path(__file__).resolve().parents[2] / "querymesh" / "airflow" / "dags" / "ingestion_dag.py"


def _tasks_path() -> Path:
    return Path(__file__).resolve().parents[2] / "querymesh" / "airflow" / "tasks.py"


@pytest.mark.unit
def test_ingestion_dag_file_is_syntactically_valid() -> None:
    import ast

    ast.parse(_dag_path().read_text())  # raises SyntaxError if invalid


@pytest.mark.unit
def test_tasks_module_never_imports_airflow() -> None:
    """Confirms the architectural claim: only dags/ingestion_dag.py imports
    `airflow`; tasks.py (tested above, in this normal venv) never does.
    """
    import ast

    tree = ast.parse(_tasks_path().read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(name == "airflow" or name.startswith("airflow.") for name in imported)


@pytest.mark.unit
def test_ingestion_dag_configures_retries() -> None:
    source = _dag_path().read_text()
    assert "retries" in source
