"""Tracing abstraction.

Rule: "Create logical tracing hierarchy" — `research_request` at the root,
with `supervisor`, `retrieval` (and its `bm25`/`vector`/`rrf` children),
`grader`, `synthesis`, `citation` nested underneath. This module provides
the `Tracer` protocol callers use to build that hierarchy without knowing
which backend(s) are active.

Both Logfire and Langfuse (current versions) are context-manager-based
span APIs, so `Tracer.span()` is a plain context manager — nested `with`
blocks naturally produce the nested hierarchy the rule asks for.

Rule: "Never log secrets. Never log API keys." — callers pass span
attributes explicitly; this module never reads or forwards raw config
(so a caller accidentally passing `settings.jina_api_key` as an attribute
is a caller bug, not something this module does for you — see
`scrub_attributes` for a defensive helper).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

from querymesh.observability.logging import get_logger

logger = get_logger(__name__)

_SENSITIVE_KEY_MARKERS = ("key", "token", "secret", "password", "credential")


def scrub_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Drop any attribute whose key looks like it might hold a secret."""
    return {
        k: v
        for k, v in attributes.items()
        if not any(marker in k.lower() for marker in _SENSITIVE_KEY_MARKERS)
    }


@runtime_checkable
class Tracer(Protocol):
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        ...


class NoOpTracer:
    """Default tracer when neither Logfire nor Langfuse is configured."""

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        yield


class LogfireTracer:
    """Wraps a configured `logfire` module."""

    def __init__(self, logfire_module: Any) -> None:
        self._logfire = logfire_module

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        with self._logfire.span(name, **scrub_attributes(attributes)):
            yield


class LangfuseTracer:
    """Wraps a configured `langfuse.Langfuse` client."""

    def __init__(self, langfuse_client: Any) -> None:
        self._client = langfuse_client

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        with self._client.start_as_current_observation(
            name=name, as_type="span", metadata=scrub_attributes(attributes)
        ):
            yield


class CompositeTracer:
    """Fans a single span out to multiple underlying tracers.

    A failure in one backend's span doesn't prevent the others from
    tracing — observability failures should never break the actual
    request.
    """

    def __init__(self, tracers: list[Tracer]) -> None:
        self._tracers = tracers

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        from contextlib import ExitStack

        with ExitStack() as stack:
            for tracer in self._tracers:
                try:
                    stack.enter_context(tracer.span(name, **attributes))
                except Exception as exc:  # noqa: BLE001 - tracing must not break the request
                    logger.warning("Tracer %s failed to start span '%s': %s", tracer, name, exc)
            yield
