"""Phase 27 unit tests: the measure/measure_many timing utility, fast and
isolated (uses asyncio.sleep for controllable, tiny durations).
"""

from __future__ import annotations

import asyncio

import pytest

from querymesh.observability.timing import LatencyResult, mean_duration_ms, measure, measure_many


@pytest.mark.unit
async def test_measure_returns_result_and_positive_duration() -> None:
    async def work() -> str:
        await asyncio.sleep(0.01)
        return "done"

    result, latency = await measure("test_op", work())

    assert result == "done"
    assert isinstance(latency, LatencyResult)
    assert latency.name == "test_op"
    assert latency.duration_ms > 0


@pytest.mark.unit
async def test_measure_captures_metadata() -> None:
    async def work() -> int:
        return 42

    _result, latency = await measure("op", work(), query="test", top_k=10)

    assert latency.metadata == {"query": "test", "top_k": 10}


@pytest.mark.unit
async def test_measure_many_runs_repeats_and_returns_one_result_each() -> None:
    call_count = {"n": 0}

    async def work() -> None:
        call_count["n"] += 1

    results = await measure_many("op", work, repeats=5)

    assert len(results) == 5
    assert call_count["n"] == 5
    assert all(r.name == "op" for r in results)
    assert all(r.duration_ms >= 0 for r in results)


@pytest.mark.unit
def test_mean_duration_ms_computes_average() -> None:
    results = [
        LatencyResult(name="op", duration_ms=10.0),
        LatencyResult(name="op", duration_ms=20.0),
        LatencyResult(name="op", duration_ms=30.0),
    ]
    assert mean_duration_ms(results) == pytest.approx(20.0)


@pytest.mark.unit
def test_mean_duration_ms_empty_list_is_zero() -> None:
    assert mean_duration_ms([]) == 0.0


@pytest.mark.unit
async def test_measure_slower_work_has_larger_duration() -> None:
    async def fast() -> None:
        pass

    async def slow() -> None:
        await asyncio.sleep(0.02)

    _r1, fast_latency = await measure("fast", fast())
    _r2, slow_latency = await measure("slow", slow())

    assert slow_latency.duration_ms > fast_latency.duration_ms
