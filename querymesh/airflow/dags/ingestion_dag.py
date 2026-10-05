"""QueryMesh continuous arXiv ingestion DAG.

Schedule
 ↓
Query arXiv
 ↓
Detect new papers
 ↓
Download PDFs
 ↓
Docling parse
 ↓
Chunk
 ↓
Embed
 ↓
Index OpenSearch

This file imports `airflow` and therefore only loads inside an Airflow
scheduler/worker process running Airflow's own isolated environment (see
`airflow/requirements.txt` — Airflow's pinned dependencies conflict with
the main application's, so they are never installed together). All actual
logic lives in `querymesh.airflow.tasks`, which has no `airflow`
dependency and is unit-tested from the main environment; this file is
intentionally thin.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

from querymesh.airflow.tasks import chunk_document, download_and_parse, embed_and_index, search_new_papers

_DEFAULT_ARGS = {
    "owner": "querymesh",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

_SEARCH_QUERY = "cat:cs.CL"  # broad default; override via Airflow Variables in a real deployment

async def _get_known_arxiv_ids() -> set[str]:
    """Return arXiv IDs already present in the QueryMesh OpenSearch index."""
    from querymesh.config import get_settings
    from querymesh.indexing.opensearch import OpenSearchConnection, build_client

    settings = get_settings()
    client = build_client(settings)

    connection = OpenSearchConnection(client=client, index_name=settings.opensearch_index, )

    try:
        response = await connection.search(
            {
                "size": 0,
                "aggs": {
                    "known_arxiv_ids": {
                        "terms": {
                            "field": "arxiv_id",
                            "size": 10000,
                        }
                    }
                },
            }
        )

        buckets = (
            response
            .get("aggregations", {})
            .get("known_arxiv_ids", {})
            .get("buckets", [])
        )

        return {str(bucket["key"]) for bucket in buckets}

    finally:
        await connection.close()

# def _search_new_papers_task(**context: Any) -> list[dict]:
#     """Search arXiv, skipping papers already ingested in a prior run.

#     `known_ids` would in a real deployment come from a persistent store
#     (e.g. an Airflow Variable, or a query against the OpenSearch index
#     itself for known arxiv_ids) — kept as an empty set here since that
#     persistence choice is a deployment detail, not core DAG logic.
#     """
#     known_ids: set[str] = set()
#     return asyncio.run(search_new_papers(query=_SEARCH_QUERY, max_results=1, known_ids=known_ids))

def _search_new_papers_task(**context: Any) -> list[dict]:
    """Search arXiv and skip papers already present in OpenSearch."""

    async def _run() -> list[dict]:
        known_ids = await _get_known_arxiv_ids()

        print(f"Found {len(known_ids)} already-ingested arXiv papers")

        papers = await search_new_papers(query=_SEARCH_QUERY, max_results=1, known_ids=known_ids, )

        print(f"Found {len(papers)} new papers to ingest")

        return papers

    return asyncio.run(_run())


def _ingest_papers_task(**context: Any) -> int:
    """Download, parse, chunk, embed, and index each newly found paper.

    One paper's failure is logged and skipped rather than failing the
    whole task — an ingestion run finding 20 papers shouldn't lose all 20
    because one PDF failed to download.
    """
    ti = context["ti"]
    papers: list[dict] = ti.xcom_pull(task_ids="search_new_papers") or []

    total_indexed = 0
    for paper in papers:
        try:
            parsed = asyncio.run(download_and_parse(paper))
            chunks = chunk_document(parsed)
            total_indexed += asyncio.run(embed_and_index(chunks))
        except Exception as exc:  # noqa: BLE001 - isolate one paper's failure from the batch
            print(f"Failed to ingest {paper.get('arxiv_id', 'unknown')}: {exc}")

    return total_indexed


with DAG(
    dag_id="querymesh_arxiv_ingestion",
    default_args=_DEFAULT_ARGS,
    description="Continuously ingest new arXiv papers into QueryMesh's index",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["querymesh", "ingestion"],
) as dag:
    search_task = PythonOperator(
        task_id="search_new_papers",
        python_callable=_search_new_papers_task,
    )
    ingest_task = PythonOperator(
        task_id="ingest_papers",
        python_callable=_ingest_papers_task,
    )

    search_task >> ingest_task
