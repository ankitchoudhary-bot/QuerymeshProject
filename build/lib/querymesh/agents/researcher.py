"""Researcher (Query Planner) agent.

Responsibilities: understand the user question, identify what research is
needed, and produce one or more concrete search queries.
"""

from __future__ import annotations

from querymesh.agents.models import ResearchPlan
from querymesh.agents.structured_output import (
    StructuredOutputError,
    parse_json_response,
)
from querymesh.errors import AgentError
from querymesh.llm.base import LLM
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = """You are a research planning assistant for scientific papers on arXiv.

Given the user's question, produce a JSON object with:
  - "search_queries": a list of 1-3 concrete search queries that would find \
relevant papers/passages
  - "reasoning": a short explanation of your plan

Respond with JSON only, no other text.

User question: {query}
"""


_REWRITE_PROMPT_TEMPLATE = """You are a research planning assistant for scientific papers on arXiv.

A previous search did not find sufficient evidence to answer the user's \
question. Propose 1-3 DIFFERENT search queries — broader, more specific, \
or using different terminology than before — to find better evidence.

Respond with JSON only:
  - "search_queries": a list of 1-3 new search queries
  - "reasoning": a short explanation of what you changed and why

User question: {query}

Previous search queries that did not find enough evidence:
{previous_queries}
"""


class ResearcherAgent:
    """Plans what to search for, given a user's question."""

    def __init__(self, llm: LLM) -> None:
        if llm is None:
            raise AgentError("ResearcherAgent requires a non-None llm")
        self.llm = llm

    async def plan(self, user_query: str) -> ResearchPlan:
        prompt = _PROMPT_TEMPLATE.format(query=user_query)
        return await self._request_plan(prompt, fallback_queries=[user_query])

    async def rewrite_queries(self, user_query: str, previous_queries: list[str]) -> ResearchPlan:
        """Propose new search queries after a previous round found
        insufficient evidence. Falls back to the original query if the
        LLM's response can't be parsed, rather than looping forever with
        no queries at all.
        """
        previous_list = "\n".join(f"- {q}" for q in previous_queries) or "(none)"
        prompt = _REWRITE_PROMPT_TEMPLATE.format(query=user_query, previous_queries=previous_list)
        return await self._request_plan(prompt, fallback_queries=previous_queries or [user_query])

    async def _request_plan(self, prompt: str, fallback_queries: list[str]) -> ResearchPlan:
        response = await self.llm.ainvoke(prompt)

        try:
            parsed = parse_json_response(response.text)
            if not isinstance(parsed, dict):
                raise StructuredOutputError("Expected a JSON object")
            queries = [str(q) for q in parsed.get("search_queries", []) if str(q).strip()]
            if not queries:
                raise StructuredOutputError("No search_queries in response")
            return ResearchPlan(search_queries=queries, reasoning=parsed.get("reasoning"))
        except StructuredOutputError as exc:
            logger.warning(
                "Researcher agent could not parse plan from LLM response (%s); "
                "falling back to prior queries.",
                exc,
            )
            return ResearchPlan(search_queries=fallback_queries, reasoning=None)
