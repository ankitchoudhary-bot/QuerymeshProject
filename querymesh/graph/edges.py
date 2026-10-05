"""Conditional routing for the research graph.

Rule: "Add a retry limit... Never create an infinite retry loop. The graph
should terminate deterministically." `route_after_supervisor` is the only
place that decision is made, and it always terminates within
`max_retries` bumps.
"""

from __future__ import annotations

from querymesh.graph.state import ResearchState

RETRY = "retry"
FINISH = "finish"


def route_after_supervisor(state: ResearchState) -> str:
    """Decide whether to retry retrieval or finish, after synthesis+citation.

    Retries only when the synthesizer flagged insufficient evidence AND the
    retry budget isn't exhausted — both conditions required, so this always
    terminates.
    """
    evidence_sufficient = state["metadata"].get("evidence_sufficient", True)
    retries_remaining = state["retry_count"] < state["max_retries"]

    if not evidence_sufficient and retries_remaining:
        return RETRY
    return FINISH
