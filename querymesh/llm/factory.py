"""LLM provider factory.

Resolves a concrete `LLM` implementation from `Settings` so callers never
construct providers directly. This is the *only* place in the codebase
allowed to import concrete provider classes.
"""

from __future__ import annotations

from querymesh.config import Settings, get_settings
from querymesh.llm.base import LLM, LLMConfigurationError
from querymesh.llm.fake import FakeLLM
from querymesh.llm.ollama import OllamaLLM

_SUPPORTED_PROVIDERS = {"fake", "ollama"}


def get_llm(settings: Settings | None = None) -> LLM:
    """Construct the configured LLM provider.

    Falls back to the `fake` provider when no provider is configured, so
    the rest of the system (agents, graph nodes) can be built and tested
    against a real object satisfying the `LLM` protocol before any real
    provider credentials exist.
    """
    settings = settings or get_settings()
    provider = (settings.llm_provider or "fake").lower()

    if provider not in _SUPPORTED_PROVIDERS:
        raise LLMConfigurationError(
            f"Unknown LLM_PROVIDER={provider!r}. Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        )

    if provider == "fake":
        return FakeLLM(model=settings.llm_model or "fake-echo-v1")

    if provider == "ollama":
        if not settings.llm_model:
            raise LLMConfigurationError("LLM_MODEL must be set when LLM_PROVIDER=ollama")
        return OllamaLLM(model=settings.llm_model)

    # Unreachable given the membership check above, but keeps type-checkers happy.
    raise LLMConfigurationError(f"Unhandled provider: {provider}")
