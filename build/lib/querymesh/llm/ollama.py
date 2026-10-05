"""Ollama LLM provider.

A real, network-calling provider (local Ollama server) implementing the
`LLM` protocol. Not exercised by the test suite (no live server in CI/dev
sandboxes) — Phase 1 tests validate the contract via `FakeLLM` instead.
This class exists so the abstraction has at least one real, non-trivial
implementation and so later phases can select it via `LLM_PROVIDER=ollama`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator

import httpx

from querymesh.llm.base import LLMChunk, LLMConfigurationError, LLMError, LLMResponse

_DEFAULT_BASE_URL = "http://localhost:11434"
_TIMEOUT_SECONDS = 60.0


class OllamaLLM:
    """LLM provider backed by a local/remote Ollama server."""

    def __init__(self, model: str, base_url: str = _DEFAULT_BASE_URL) -> None:
        if not model:
            raise LLMConfigurationError("OllamaLLM requires a model name")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def invoke(self, prompt: str) -> LLMResponse:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
                resp = client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        return LLMResponse(
            text=data.get("response", ""),
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=(data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
            finish_reason="stop" if data.get("done") else None,
            raw=data,
        )

    async def ainvoke(self, prompt: str) -> LLMResponse:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                resp = await client.post(f"{self.base_url}/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        return LLMResponse(
            text=data.get("response", ""),
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=(data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
            finish_reason="stop" if data.get("done") else None,
            raw=data,
        )

    def stream(self, prompt: str) -> Iterator[LLMChunk]:
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
                with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        yield LLMChunk(
                            text=data.get("response", ""),
                            model=self.model,
                            is_final=bool(data.get("done")),
                            raw=data,
                        )
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama streaming request failed: {exc}") from exc

    async def astream(self, prompt: str) -> AsyncIterator[LLMChunk]:
        payload = {"model": self.model, "prompt": prompt, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        yield LLMChunk(
                            text=data.get("response", ""),
                            model=self.model,
                            is_final=bool(data.get("done")),
                            raw=data,
                        )
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama streaming request failed: {exc}") from exc
