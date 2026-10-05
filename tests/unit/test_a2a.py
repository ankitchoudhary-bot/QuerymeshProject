"""Phase 17 tests: AgentMessage envelope construction and A2AClient
dispatch — success, unknown recipient, timeout+retry, and loop prevention.
"""

from __future__ import annotations

import asyncio

import pytest

from querymesh.a2a import (
    A2AAgentNotFoundError,
    A2AClient,
    A2ALoopDetectedError,
    AgentMessage,
    build_error,
    build_request,
    build_response,
)


# --- Protocol envelope ---------------------------------------------------


@pytest.mark.unit
def test_build_request_has_all_required_fields() -> None:
    msg = build_request(sender="a", recipient="b", task="do_thing", payload={"x": 1})

    assert msg.sender == "a"
    assert msg.recipient == "b"
    assert msg.task == "do_thing"
    assert msg.payload == {"x": 1}
    assert msg.status == "request"
    assert msg.message_id
    assert msg.correlation_id
    assert msg.timestamp is not None
    assert msg.hop_count == 0


@pytest.mark.unit
def test_build_response_swaps_sender_and_recipient() -> None:
    request = build_request(sender="a", recipient="b", task="t", payload={})
    response = build_response(request, payload={"result": "ok"})

    assert response.sender == "b"
    assert response.recipient == "a"
    assert response.status == "response"
    assert response.correlation_id == request.correlation_id
    assert response.payload == {"result": "ok"}


@pytest.mark.unit
def test_build_error_carries_error_message() -> None:
    request = build_request(sender="a", recipient="b", task="t", payload={})
    error = build_error(request, "something broke")

    assert error.status == "error"
    assert error.error == "something broke"
    assert error.correlation_id == request.correlation_id


@pytest.mark.unit
def test_each_message_has_unique_id() -> None:
    m1 = build_request(sender="a", recipient="b", task="t", payload={})
    m2 = build_request(sender="a", recipient="b", task="t", payload={})
    assert m1.message_id != m2.message_id


# --- A2AClient: success path ------------------------------------------


@pytest.mark.unit
async def test_client_dispatches_to_registered_handler() -> None:
    client = A2AClient()

    async def handler(payload: dict) -> dict:
        return {"doubled": payload["value"] * 2}

    client.register_handler("worker", handler)

    response = await client.send(sender="supervisor", recipient="worker", task="double", payload={"value": 5})

    assert response.status == "response"
    assert response.payload == {"doubled": 10}
    assert response.sender == "worker"
    assert response.recipient == "supervisor"


# --- Unknown recipient ---------------------------------------------------


@pytest.mark.unit
async def test_client_raises_for_unknown_recipient() -> None:
    client = A2AClient()

    with pytest.raises(A2AAgentNotFoundError):
        await client.send(sender="a", recipient="nonexistent", task="t", payload={})


# --- Timeout + retry ----------------------------------------------------


@pytest.mark.unit
async def test_client_retries_transient_failure_then_succeeds() -> None:
    client = A2AClient(max_retries=2)
    attempts = {"count": 0}

    async def flaky_handler(payload: dict) -> dict:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient failure")
        return {"ok": True}

    client.register_handler("flaky", flaky_handler)

    response = await client.send(sender="a", recipient="flaky", task="t", payload={})

    assert attempts["count"] == 3
    assert response.status == "response"
    assert response.payload == {"ok": True}


@pytest.mark.unit
async def test_client_returns_error_message_after_exhausting_retries() -> None:
    client = A2AClient(max_retries=1)

    async def always_fails(payload: dict) -> dict:
        raise RuntimeError("permanent failure")

    client.register_handler("broken", always_fails)

    response = await client.send(sender="a", recipient="broken", task="t", payload={})

    assert response.status == "error"
    assert "permanent failure" in response.error


@pytest.mark.unit
async def test_client_returns_timeout_error_after_exhausting_retries() -> None:
    client = A2AClient(timeout_seconds=0.05, max_retries=1)

    async def slow_handler(payload: dict) -> dict:
        await asyncio.sleep(1.0)
        return {"never": "reached"}

    client.register_handler("slow", slow_handler)

    response = await client.send(sender="a", recipient="slow", task="t", payload={})

    assert response.status == "error"
    assert "did not respond" in response.error


# --- Loop prevention -----------------------------------------------------


@pytest.mark.unit
async def test_client_raises_loop_detected_when_max_hops_exceeded() -> None:
    client = A2AClient(max_hops=3)

    async def handler(payload: dict) -> dict:
        return {"ok": True}

    client.register_handler("worker", handler)

    with pytest.raises(A2ALoopDetectedError):
        await client.send(sender="a", recipient="worker", task="t", payload={}, hop_count=3)


@pytest.mark.unit
async def test_client_allows_hop_count_under_limit() -> None:
    client = A2AClient(max_hops=3)

    async def handler(payload: dict) -> dict:
        return {"ok": True}

    client.register_handler("worker", handler)

    response = await client.send(sender="a", recipient="worker", task="t", payload={}, hop_count=2)
    assert response.status == "response"
