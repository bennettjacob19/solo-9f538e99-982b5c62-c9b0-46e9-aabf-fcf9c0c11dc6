"""Batching, ordering, deadline and failure-handling tests for the gateway."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gateway.app.batcher import BatchScheduler
from gateway.app.errors import ClientDisconnected, RequestTimeout
from tests.conftest import FakeBackend, gateway_client, post, vector_for


def assert_embeddings(response: httpx.Response, texts: list[str]) -> None:
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["data"]) == len(texts)
    assert [item["index"] for item in body["data"]] == list(range(len(texts)))
    for item, text in zip(body["data"], texts):
        assert item["embedding"] == vector_for(text)


@pytest.mark.asyncio
async def test_single_request_round_trip_and_validation() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        response = await post(client, ["alpha"])
        assert_embeddings(response, ["alpha"])
        assert response.json()["batch_size"] == 1

        too_many = await post(client, ["a", "b", "c", "d", "e"])
        assert too_many.status_code == 422
        assert too_many.json()["error"]["code"] == "validation_error"
        empty = await post(client, [])
        assert empty.status_code == 422
        bad_timeout = await post(client, ["a"], timeout_ms=600001)
        assert bad_timeout.status_code == 422
        bad_type = await client.post("/v1/embeddings", json={"input": [1, 2]})
        assert bad_type.status_code == 422
        assert bad_type.json()["error"]["code"] == "validation_error"
        zero_timeout = await post(client, ["a"], timeout_ms=0)
        assert zero_timeout.status_code == 422
        assert zero_timeout.json()["error"]["code"] == "validation_error"
        assert backend.batch_sizes == [1]


@pytest.mark.asyncio
async def test_concurrent_requests_merge_to_eight_and_keep_order() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        groups = [
            ["a1", "a2", "a3", "a4"],
            ["b1", "b2", "b3", "b4"],
            ["c1", "c2"],
        ]
        responses = await asyncio.gather(
            *[post(client, texts) for texts in groups]
        )

        for response, texts in zip(responses, groups):
            assert_embeddings(response, texts)

        # Three concurrent requests, first-come texts merged into batches of
        # at most 8 without splitting a request.
        assert backend.batch_sizes == [8, 2]
        assert sorted(backend.sent_texts) == sorted(
            [text for group in groups for text in group]
        )
        # Callers from the same first batch report the same batch id.
        assert responses[0].json()["batch_id"] == responses[1].json()["batch_id"]
        assert responses[0].json()["batch_size"] == 8
        assert responses[2].json()["batch_size"] == 2
        assert backend.max_active == 1


@pytest.mark.asyncio
async def test_window_does_not_split_when_second_request_too_large() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        three = ["x1", "x2", "x3"]
        four = ["y1", "y2", "y3", "y4"]
        responses = await asyncio.gather(post(client, three), post(client, four))
        assert_embeddings(responses[0], three)
        assert_embeddings(responses[1], four)
        # 3 + 4 = 7 leaves room for 1, but a request is never split, so the
        # second request is not merged when a third request of size 2 arrives.
        # Here only two requests exist, so they do merge.
        assert backend.batch_sizes == [7]

        two = ["z1", "z2"]
        r1, r2 = await asyncio.gather(
            post(client, ["q1", "q2", "q3", "q4"]),
            post(client, two),
        )
        assert_embeddings(r1, ["q1", "q2", "q3", "q4"])
        assert_embeddings(r2, two)
        assert backend.batch_sizes == [7, 6]


@pytest.mark.asyncio
async def test_idle_backend_waits_full_window_then_sends() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        loop = asyncio.get_running_loop()
        started = loop.time()
        response = await post(client, ["solo"])
        elapsed = loop.time() - started
        assert response.status_code == 200
        # Window is 20 ms; allow scheduling slack but require the wait.
        assert 0.015 <= elapsed < 0.5
        assert backend.batch_sizes == [1]


@pytest.mark.asyncio
async def test_timed_out_pending_request_is_removed_from_batch() -> None:
    backend = FakeBackend()
    async with gateway_client(backend) as (_, client):
        results = await asyncio.gather(
            post(client, ["expired"], timeout_ms=5),
            post(client, ["survivor"], timeout_ms=5000),
        )
        assert results[0].status_code == 504
        assert results[0].json()["error"]["code"] == "request_timeout"
        assert_embeddings(results[1], ["survivor"])
        assert backend.sent_texts == ["survivor"]
        assert backend.batch_sizes == [1]


@pytest.mark.asyncio
async def test_backend_error_fails_all_waiters_then_recovers() -> None:
    backend = FakeBackend(status=503)
    async with gateway_client(backend) as (_, client):
        responses = await asyncio.gather(
            post(client, ["e1"]),
            post(client, ["e2", "e3"]),
        )
        for response in responses:
            assert response.status_code == 502
            assert response.json()["error"]["code"] == "backend_error"
        assert len(backend.calls) == 1

        stats = (await client.get("/stats")).json()
        assert stats["pending_requests"] == 0
        assert stats["in_flight"] is False

        backend.status = 200
        recovered = await post(client, ["ok"])
        assert_embeddings(recovered, ["ok"])
        assert backend.batch_sizes == [3, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["malformed", "bad_coalesce", "short", "duplicate_index"]
)
async def test_invalid_backend_payloads_fail_batch_without_hanging(
    mode: str,
) -> None:
    backend = FakeBackend(mode=mode)
    async with gateway_client(backend) as (_, client):
        responses = await asyncio.gather(
            post(client, ["m1", "m2"]), post(client, ["m3"])
        )
        for response in responses:
            assert response.status_code == 502
        assert len(backend.calls) == 1

        backend.mode = "ok"
        assert_embeddings(await post(client, ["after"]), ["after"])


@pytest.mark.asyncio
async def test_single_flight_waits_for_late_response_even_if_all_callers_left() -> None:
    backend = FakeBackend(delay=0.25)
    async with gateway_client(backend) as (_, client):
        first = await post(client, ["late"], timeout_ms=50)
        assert first.status_code == 504

        # The backend is still busy with the abandoned batch.  A new request
        # must wait for it and must never run concurrently with it.
        second = await post(client, ["next"], timeout_ms=5000)
        assert_embeddings(second, ["next"])

        assert len(backend.calls) == 2
        assert backend.calls[1].start >= (backend.calls[0].end or float("inf"))
        assert backend.max_active == 1
        assert backend.sent_texts == ["late", "next"]


@pytest.mark.asyncio
async def test_late_vectors_are_not_delivered_to_expired_request() -> None:
    backend = FakeBackend(delay=0.2)
    async with gateway_client(backend) as (_, client):
        response = await post(client, ["ghost"], timeout_ms=60)
        assert response.status_code == 504
        # Wait until the late backend response has definitely arrived and the
        # scheduler moved on.
        await asyncio.sleep(0.3)
        stats = (await client.get("/stats")).json()
        assert stats["completed_requests"] == 0
        assert stats["timed_out_requests"] == 1
        assert stats["pending_requests"] == 0
        assert stats["in_flight"] is False
        assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_timed_out_request_with_queued_work_only_sends_live_requests() -> None:
    backend = FakeBackend(delay=0.2)
    async with gateway_client(backend) as (_, client):
        # Occupy the backend so the following requests pile up while it is
        # busy, then let the short-deadline request expire while pending.
        blocker_task = asyncio.create_task(post(client, ["blocker"], timeout_ms=5000))
        await asyncio.sleep(0.05)
        results = await asyncio.gather(
            post(client, ["short"], timeout_ms=40),
            post(client, ["long"], timeout_ms=5000),
        )
        assert_embeddings(await blocker_task, ["blocker"])
        assert results[0].status_code == 504
        assert_embeddings(results[1], ["long"])
        assert backend.sent_texts == ["blocker", "long"]
        assert backend.max_active == 1


class _FakeRequest:
    def __init__(self, disconnect_after: float | None) -> None:
        self._disconnect_after = disconnect_after
        self._started = asyncio.get_running_loop().time()

    async def is_disconnected(self) -> bool:
        if self._disconnect_after is None:
            return False
        return asyncio.get_running_loop().time() - self._started >= self._disconnect_after


@pytest.mark.asyncio
async def test_disconnect_removes_pending_request_and_ignores_late_result() -> None:
    backend = FakeBackend(delay=0.15)
    settings = __import__("tests.conftest", fromlist=["make_settings"]).make_settings()
    client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
    scheduler = BatchScheduler(settings, client)
    await scheduler.start()
    try:
        loop = asyncio.get_running_loop()
        with pytest.raises(ClientDisconnected):
            await scheduler.submit(
                request_id="gone",
                texts=["never-sent"],
                model="fake-model",
                deadline=loop.time() + 5,
                request=_FakeRequest(disconnect_after=0.0),
            )
        await asyncio.sleep(0.05)
        stats = scheduler.stats()
        assert stats["pending_requests"] == 0
        assert stats["disconnected_requests"] == 1
        assert backend.calls == []

        # A request that is already on the wire when its deadline expires must
        # release its caller immediately; its (late) vectors are discarded.
        with pytest.raises(RequestTimeout):
            await scheduler.submit(
                request_id="inflight-gone",
                texts=["inflight"],
                model="fake-model",
                deadline=loop.time() + 0.03,
                request=_FakeRequest(disconnect_after=None),
            )
        await asyncio.sleep(0.2)
        stats = scheduler.stats()
        assert stats["completed_requests"] == 0
        assert stats["timed_out_requests"] == 1
        assert backend.calls and backend.calls[-1].texts == ["inflight"]
        assert backend.calls[-1].end is not None
    finally:
        await scheduler.close()
        await client.aclose()
