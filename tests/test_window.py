"""Gateway must honour an idle 20 ms merge window and never exceed 8 texts."""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.conftest import FakeBackend, gateway_client, post


@pytest.mark.asyncio
async def test_backlog_is_merged_immediately_without_extra_window() -> None:
    backend = FakeBackend(delay=0.1)
    async with gateway_client(backend) as (_, client):
        await post(client, ["warm"])  # warm the connection
        started = time.monotonic()
        blocker = asyncio.create_task(post(client, ["hold"]))
        # Wait until the blocker batch is on the wire; the following requests
        # therefore pile up while the backend is busy and their merge window
        # elapses before the backend becomes free again.
        await asyncio.sleep(0.05)
        pending = [
            asyncio.create_task(post(client, ["p1", "p2"])),
            asyncio.create_task(post(client, ["p3", "p4"])),
            asyncio.create_task(post(client, ["p5", "p6"])),
            asyncio.create_task(post(client, ["p7", "p8"])),
            asyncio.create_task(post(client, ["p9"])),
            asyncio.create_task(post(client, ["p10"])),
        ]
        responses = await asyncio.gather(blocker, *pending)
        elapsed = time.monotonic() - started
        assert all(r.status_code == 200 for r in responses)
        # Backlog: 10 queued texts are merged into 8 + 2.  A collector that
        # re-armed a 20 ms window (or broke out of the batch as soon as the
        # first request's window had elapsed) would send them one at a time.
        assert backend.batch_sizes[2:] == [8, 2]
        assert backend.max_active == 1
        # Two backend calls of 100 ms each for the backlog, not six.
        assert elapsed < 0.5


@pytest.mark.asyncio
async def test_never_exceeds_eight_texts_per_batch() -> None:
    backend = FakeBackend(delay=0.02)
    async with gateway_client(backend) as (_, client):
        groups = [[f"t{i}-{j}" for j in range(4)] for i in range(10)]
        responses = await asyncio.gather(
            *[post(client, group) for group in groups]
        )
        for group, response in zip(groups, responses):
            assert response.status_code == 200
            assert len(response.json()["data"]) == 4
        assert all(size <= 8 for size in backend.batch_sizes)
        assert max(backend.batch_sizes) == 8
        assert backend.max_active == 1


@pytest.mark.asyncio
async def test_first_queued_request_window_bounds_latency() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        started = time.monotonic()
        response = await post(client, ["alone"], timeout_ms=5000)
        elapsed = time.monotonic() - started
        assert response.status_code == 200
        assert 0.015 <= elapsed < 0.3
