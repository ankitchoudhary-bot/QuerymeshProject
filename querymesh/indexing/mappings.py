"""OpenSearch index mapping for the papers/chunks index.

Supports:
  - text fields (full-text, BM25-searchable): title, section, text
  - keyword fields (exact match / filtering): arxiv_id, chunk_id, categories
  - date fields: published, updated
  - numeric fields: chunk_index, page
  - a dense_vector-equivalent (OpenSearch `knn_vector`) field for embeddings
  - a free-form metadata object for anything not worth a dedicated field
"""

from __future__ import annotations

# Jina embedding dimensions vary by model; this default matches
# jina-embeddings-v3's common 1024-dim output. Overridable per index.
DEFAULT_EMBEDDING_DIM = 1024


def build_index_settings(embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> dict:
    """Return the full index body (settings + mappings) for paper chunks."""
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "properties": {
                # --- keyword / identity fields -----------------------------
                "chunk_id": {"type": "keyword"},
                "arxiv_id": {"type": "keyword"},
                "categories": {"type": "keyword"},
                # --- text fields (BM25) -------------------------------------
                "title": {"type": "text"},
                "section": {"type": "text"},
                "text": {"type": "text"},
                "abstract": {"type": "text"},
                # Chunk type
                "chunk_type": {"type": "keyword"},
                # Asset reference
                "image_path": {"type": "keyword"},
                "table_markdown": {"type": "text"},
                # --- date fields ---------------------------------------------
                "published": {"type": "date"},
                "updated": {"type": "date"},
                # --- numeric fields --------------------------------------------
                "chunk_index": {"type": "integer"},
                "page": {"type": "integer"},
                # --- vector field ------------------------------------------------
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
                # --- free-form metadata ---------------------------------------
                "metadata": {"type": "object", "enabled": True},
            }
        },
    }




# {
#   "settings": {
#     "analysis": {
#       "filter": {
#         "english_stemmer": {
#           "type": "stemmer",
#           "language": "english"
#         }
#       },
#       "analyzer": {
#         "scientific_english": {
#           "type": "custom",
#           "tokenizer": "standard",
#           "filter": [
#             "lowercase",
#             "english_stemmer"
#           ]
#         }
#       }
#     }
#   },
#   "mappings": {
#     "properties": {
#       "abstract": {
#         "type": "text",
#         "analyzer": "scientific_english"
#       },
#       "arxiv_id": {
#         "type": "keyword"
#       },
#       "categories": {
#         "type": "keyword"
#       },
#       "chunk_id": {
#         "type": "keyword"
#       },
#       "chunk_index": {
#         "type": "integer"
#       },
#       "chunk_type": {
#         "type": "keyword"
#       },
#       "embedding": {
#         "type": "knn_vector",
#         "dimension": 1024,
#         "method": {
#           "engine": "lucene",
#           "space_type": "cosinesimil",
#           "name": "hnsw",
#           "parameters": {}
#         }
#       },
#       "image_path": {
#         "type": "keyword"
#       },
#       "metadata": {
#         "type": "object"
#       },
#       "page": {
#         "type": "integer"
#       },
#       "published": {
#         "type": "date"
#       },
#       "section": {
#         "type": "text",
#         "analyzer": "scientific_english"
#       },
#       "table_markdown": {
#         "type": "text",
#         "analyzer": "scientific_english"
#       },
#       "text": {
#         "type": "text",
#         "analyzer": "scientific_english"
#       },
#       "title": {
#         "type": "text",
#         "analyzer": "scientific_english"
#       },
#       "updated": {
#         "type": "date"
#       }
#     }
#   }
# }