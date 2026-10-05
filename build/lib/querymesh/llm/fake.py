"""Fake LLM provider.

Used for tests and local development so the graph/agents built in later
phases can be exercised without a real provider or network access. This
is a first-class provider (selectable via `LLM_PROVIDER=fake`), not a
mock — it fully implements the `LLM` protocol.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

from querymesh.llm.base import LLMChunk, LLMResponse


class FakeLLM:
    """Deterministic in-memory LLM used for tests and offline development.

    By default it echoes the prompt back prefixed with a marker, which is
    enough to assert the interface contract without depending on wording
    from a real model.
    """

    def __init__(self, model: str = "fake-echo-v1", response_text: str | None = None) -> None:
        self.model = model
        self._response_text = response_text

    def _build_text(self, prompt: str) -> str:
        return self._response_text or f"[fake-response] {prompt}"

    def invoke(self, prompt: str) -> LLMResponse:
        text = self._build_text(prompt)
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            total_tokens=len(prompt.split()) + len(text.split()),
            finish_reason="stop",
        )

    async def ainvoke(self, prompt: str) -> LLMResponse:
        return self.invoke(prompt)

    def stream(self, prompt: str) -> Iterator[LLMChunk]:
        text = self._build_text(prompt)
        words = text.split(" ")
        for i, word in enumerate(words):
            is_final = i == len(words) - 1
            yield LLMChunk(text=word + ("" if is_final else " "), model=self.model, is_final=is_final)

    async def astream(self, prompt: str) -> AsyncIterator[LLMChunk]:
        for chunk in self.stream(prompt):
            yield chunk
