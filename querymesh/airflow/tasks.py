"""Airflow task functions — pure querymesh logic, no `airflow` import.

Kept deliberately separate from `dags/ingestion_dag.py`: Apache Airflow
2.10.x pins older `opentelemetry`/`httpx`/`cryptography`/`typing-extensions`
than this project's application stack needs (pydantic 2.13, logfire,
langfuse, fastmcp all require newer versions) — installing Airflow into
the same environment produces real, unresolvable dependency conflicts.

The standard, production-correct fix (per project rule 49: "adapt the
implementation, keep the architecture stable") is to run Airflow in its
own isolated environment (see `airflow/requirements.txt`), never sharing
a venv with the application. These task functions have zero dependency on
`airflow` itself, so they're importable and unit-tested from the main
querymesh environment exactly like every other module in this project;
only `dags/ingestion_dag.py` (loaded by the Airflow scheduler in its own
env) imports `airflow` and wires these functions into operators.

Idempotency:
  - `search_new_papers` filters out `known_ids`, so re-running the DAG
    never reprocesses papers already ingested.
  - `PdfDownloader` (Phase 3) already skips re-downloading an existing
    file.
  - Chunk IDs are deterministic (Phase 4) and indexing is upsert-by-ID
    (Phase 5), so re-indexing the same paper overwrites rather than
    duplicates.
"""

from __future__ import annotations

from querymesh.embeddings import get_embedder
from querymesh.indexing import ChunkIndexer, OpenSearchConnection, build_client
from querymesh.ingestion import (
    ArxivClient,
    DoclingParser,
    IngestionPipeline,
    PaperChunker,
    PdfDownloader,
)
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.ingestion.document_models import ParsedDocument
from querymesh.ingestion.models import PaperMetadata
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_STORAGE_DIR = "./data/pdfs"
_DEFAULT_INDEX_NAME = "querymesh-papers"


async def search_new_papers(query: str, max_results: int = 1, known_ids: set[str] | None = None) -> list[dict]:
    """Search arXiv and return only papers not already known.

    Returns plain dicts (not `PaperMetadata`) since this crosses an
    Airflow XCom boundary, which serializes task outputs to JSON.
    """
    known_ids = known_ids or set()
    async with ArxivClient() as client:
        papers = await client.search(query, max_results=max_results)

    new_papers = [p for p in papers if p.arxiv_id not in known_ids]
    logger.info("search_new_papers(%r): %d new of %d total", query, len(new_papers), len(papers))
    return [p.model_dump(mode="json") for p in new_papers]


async def download_and_parse(paper: dict, storage_dir: str = _DEFAULT_STORAGE_DIR) -> dict:
    """Download a paper's PDF (skipped if already on disk) and parse it."""
    downloader = PdfDownloader(storage_dir=storage_dir)
    parser = DoclingParser()
    pipeline = IngestionPipeline(downloader=downloader, parser=parser)

    try:
        parsed = await pipeline.process(PaperMetadata(**paper))
    finally:
        await downloader.aclose()

    return parsed.model_dump(mode="json")


def chunk_document(parsed_document: dict, max_chars: int = 1200) -> list[dict]:
    """Chunk a parsed document. Synchronous — chunking is pure CPU work."""
    chunker = PaperChunker(max_chars=max_chars)
    chunks = chunker.chunk(ParsedDocument(**parsed_document))
    return [c.model_dump(mode="json") for c in chunks]


# async def embed_and_index(
#     chunks: list[dict], index_name: str = _DEFAULT_INDEX_NAME
# ) -> int:
#     """Embed and bulk-index a batch of chunks. Returns the count indexed."""
#     if not chunks:
#         return 0

#     client = build_client()
#     connection = OpenSearchConnection(client=client, index_name=index_name)
#     indexer = ChunkIndexer(connection=connection, embedder=get_embedder())

#     try:
#         return await indexer.index_chunks([PaperChunk(**c) for c in chunks])
#     finally:
#         await connection.close()


async def embed_and_index(chunks: list[dict], index_name: str = _DEFAULT_INDEX_NAME) -> int:
    """Embed and bulk-index a batch of chunks. Returns the count indexed."""
    if not chunks:
        return 0

    client = build_client()
    connection = OpenSearchConnection(client=client, index_name=index_name)

    try:
        await connection.create_index(embedding_dim=1024)

        indexer = ChunkIndexer(connection=connection, embedder=get_embedder(), )

        return await indexer.index_chunks([PaperChunk(**c) for c in chunks])
    finally:
        await connection.close()