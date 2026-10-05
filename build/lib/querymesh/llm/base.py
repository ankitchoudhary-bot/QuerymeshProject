"""LLM abstraction layer.

QueryMesh rule: "The rest of the application should depend on this
abstraction rather than directly constructing an LLM everywhere."

Every concrete provider (Ollama, OpenAI-compatible, Anthropic, etc. — added
in later phases as needed) implements the `LLM` protocol below. Agents,
graph nodes, and services must only ever type-hint against `LLM`, never
against a concrete provider class.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from querymesh.errors import QueryMeshError


class LLMResponse(BaseModel):
    """A complete (non-streaming) LLM response."""

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    raw: dict = Field(default_factory=dict)


class LLMChunk(BaseModel):
    """A single streamed token/delta from an LLM."""

    text: str
    model: str
    is_final: bool = False
    raw: dict = Field(default_factory=dict)


@runtime_checkable
class LLM(Protocol):
    """Provider-agnostic LLM interface.

    Concrete implementations live in `querymesh/llm/<provider>.py` and are
    constructed exclusively through `querymesh.llm.factory.get_llm()`.
    """

    model: str

    def invoke(self, prompt: str) -> LLMResponse:
        """Synchronous, non-streaming completion."""
        ...

    async def ainvoke(self, prompt: str) -> LLMResponse:
        """Asynchronous, non-streaming completion."""
        ...

    def stream(self, prompt: str) -> Iterator[LLMChunk]:
        """Synchronous, streaming completion."""
        ...

    async def astream(self, prompt: str) -> AsyncIterator[LLMChunk]:
        """Asynchronous, streaming completion."""
        ...


class LLMError(QueryMeshError):
    """Base error for LLM-layer failures. Retryable by default — most LLM
    errors (rate limits, transient network issues) are.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message, retryable=retryable)


class LLMConfigurationError(LLMError):
    """Raised when an LLM provider is misconfigured (e.g. missing API key).
    Never retryable — retrying without fixing the config just fails again.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
