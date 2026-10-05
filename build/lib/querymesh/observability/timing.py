"""Latency measurement utility.

A small, dependency-free timer used by the performance benchmarks
(Phase 27) and available generally wherever a call site wants a duration
without pulling in the full tracing machinery. Complements (doesn't
replace) `Tracer.span()` — spans carry duration implicitly for
Logfire/Langfuse; `measure()` is for when you want the number directly,
e.g. to compare or aggregate in a report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


@dataclass
class LatencyResult:
    name: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


async def measure(
    name: str, coro: Awaitable[T], **metadata: Any
) -> tuple[T, LatencyResult]:
    """Await `coro`, returning its result alongside a `LatencyResult`."""
    start = time.perf_counter()
    result = await coro
    duration_ms = (time.perf_counter() - start) * 1000
    return result, LatencyResult(name=name, duration_ms=duration_ms, metadata=metadata)


async def measure_many(
    name: str, func: Callable[[], Awaitable[T]], repeats: int, **metadata: Any
) -> list[LatencyResult]:
    """Run `func()` `repeats` times, returning one `LatencyResult` per run.

    Useful for getting a stable mean/spread rather than a single sample.
    """
    results: list[LatencyResult] = []
    for _ in range(repeats):
        _value, latency = await measure(name, func(), **metadata)
        results.append(latency)
    return results


def mean_duration_ms(results: list[LatencyResult]) -> float:
    if not results:
        return 0.0
    return sum(r.duration_ms for r in results) / len(results)
