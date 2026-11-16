"""Race-condition regression tests for the scheduler."""

from __future__ import annotations

import asyncio

import pytest

from gateway.app.batcher import BatchScheduler
from gateway.app.errors import GatewayError, RequestTimeout
from tests.conftest import FakeBackend, gateway_client, make_settings, post


@pytest.mark.asyncio
async def test_expiry_detected_by_dispatcher_returns_504_not_cancel() -> None:
    """If the collector drops an expired entry first, the caller must still
    receive a 504 instead of an unexpected cancellation."""
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        results = await asyncio.gather(
            post(client, ["e"], timeout_ms=1),
            post(client, ["keep"], timeout_ms=5000),
        )
        assert results[0].status_code == 504
        assert results[1].status_code == 200


@pytest.mark.asyncio
async def test_backend_never_receives_more_than_two_requests() -> None:
    """Regression: abandoned in-flight batch must still gate the next one."""
    backend = FakeBackend(delay=0.12)
    async with gateway_client(backend) as (_, client):
        await asyncio.gather(
            post(client, ["a"], timeout_ms=30),
            post(client, ["b"], timeout_ms=30),
            post(client, ["c"], timeout_ms=30),
        )
        backend.delay = 0.0
        await post(client, ["d"])
        assert backend.max_active == 1
        assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_close_releases_queued_waiters() -> None:
    backend = FakeBackend(delay=0.15)
    settings = make_settings()
    import httpx

    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        loop = asyncio.get_running_loop()
        inflight = asyncio.create_task(
            scheduler.submit(
                request_id="inflight",
                texts=["x"],
                model="fake-model",
                deadline=loop.time() + 10,
            )
        )
        await asyncio.sleep(0.03)  # let the first batch reach the backend
        queued = asyncio.create_task(
            scheduler.submit(
                request_id="queued",
                texts=["y"],
                model="fake-model",
                deadline=loop.time() + 10,
            )
        )
        await asyncio.sleep(0.01)

        await scheduler.close()

        # The in-flight batch is drained; the queued caller is released with
        # a failure rather than hanging forever.
        result = await asyncio.wait_for(inflight, timeout=2)
        assert len(result.vectors) == 1
        with pytest.raises(GatewayError):
            await queued
        assert not scheduler._in_flight
        assert scheduler.stats()["pending_requests"] == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_timeout_before_queueing_is_504() -> None:
    backend = FakeBackend()
    settings = make_settings()
    import httpx

    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        loop = asyncio.get_running_loop()
        with pytest.raises(RequestTimeout):
            await scheduler.submit(
                request_id="late",
                texts=["x"],
                model="fake-model",
                deadline=loop.time() - 1,
            )
        assert backend.calls == []
    finally:
        await scheduler.close()
        await client.aclose()
