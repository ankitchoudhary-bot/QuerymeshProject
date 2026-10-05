"""Resolves the configured `Tracer` from `Settings`.

Mirrors the LLM/Embedder factories: `get_tracer()` always returns a usable
`Tracer`, falling back to `NoOpTracer` when neither backend is configured,
and composing both via `CompositeTracer` when both are.
"""

from __future__ import annotations

from querymesh.config import Settings, get_settings
from querymesh.observability.langfuse_setup import configure_langfuse
from querymesh.observability.logfire_setup import configure_logfire
from querymesh.observability.tracing import CompositeTracer, LangfuseTracer, LogfireTracer, NoOpTracer, Tracer


def get_tracer(settings: Settings | None = None) -> Tracer:
    settings = settings or get_settings()

    tracers: list[Tracer] = []

    logfire_module = configure_logfire(settings)
    if logfire_module is not None:
        tracers.append(LogfireTracer(logfire_module))

    langfuse_client = configure_langfuse(settings)
    if langfuse_client is not None:
        tracers.append(LangfuseTracer(langfuse_client))

    if not tracers:
        return NoOpTracer()
    if len(tracers) == 1:
        return tracers[0]
    return CompositeTracer(tracers)
