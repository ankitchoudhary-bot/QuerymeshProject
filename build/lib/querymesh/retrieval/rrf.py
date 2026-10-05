"""Reciprocal Rank Fusion (RRF).

RRF(d) = sum over result lists containing d of 1 / (k + rank(d))

Generic over any number of ranked result lists — used here to fuse BM25 +
vector, but doesn't assume exactly two.
"""

from __future__ import annotations

from querymesh.retrieval.models import RetrievedDocument

_DEFAULT_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedDocument]],
    k: int = _DEFAULT_K,
) -> list[RetrievedDocument]:
    """Fuse multiple ranked result lists into one, ranked by RRF score.

    Documents are matched across lists by `chunk_id`. A document missing a
    `chunk_id` (shouldn't normally happen once chunks are indexed) is
    dropped rather than silently mis-fused.
    """
    scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    for results in result_lists:
        for rd in results:
            chunk_id = rd.chunk_id
            if not chunk_id:
                continue
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rd.rank)
            doc_lookup.setdefault(chunk_id, rd.document)

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    return [
        RetrievedDocument(
            document=doc_lookup[chunk_id],
            score=scores[chunk_id],
            rank=rank,
            retrieval_method="hybrid_rrf",
        )
        for rank, chunk_id in enumerate(ranked_ids, start=1)
    ]
