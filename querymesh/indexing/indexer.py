"""Ties `Embedder` + `OpenSearchConnection` together: embeds chunk text and
bulk-indexes the resulting documents (with their `embedding` vectors) into
the index Phase 5's mapping defines.
"""

from __future__ import annotations

from querymesh.embeddings.base import Embedder
from querymesh.indexing.opensearch import OpenSearchConnection
from querymesh.ingestion.chunk_models import PaperChunk
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


class ChunkIndexer:
    """Embeds `PaperChunk`s and bulk-indexes them into OpenSearch."""

    def __init__(self, connection: OpenSearchConnection, embedder: Embedder) -> None:
        self.connection = connection
        self.embedder = embedder

    async def index_chunks(self, chunks: list[PaperChunk]) -> int:
        """Embed and index a batch of chunks. Returns the number indexed."""
        if not chunks:
            return 0

        embeddings = await self.embedder.embed_documents([c.text for c in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks")

        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = chunk.model_dump()
            doc["embedding"] = embedding
            documents.append(doc)

        return await self.connection.bulk_index(documents, id_field="chunk_id")
