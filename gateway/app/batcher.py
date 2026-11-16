"""Single-flight batching scheduler.

Design notes
------------
* There is exactly one dispatcher task, so at most one backend batch is in
  flight.  A new batch is not started until the previous response (or error)
  has been fully processed, even when every caller of that batch has already
  timed out or disconnected.
* A pending request is never split across batches.  Requests are merged FIFO
  while the accumulated text count stays at or below ``max_texts_per_batch``
  and the model matches.
* The first pending request opens a ``batch_window_ms`` window.  The window
  starts at the moment the request became pending, so requests that piled up
  while the backend was busy are dispatched immediately afterwards.
* A request whose deadline expires (or whose HTTP client disconnects) while
  still pending is removed from the queue and marked cancelled.  A response
  that arrives after a request was abandoned is dropped instead of being
  delivered to that caller.
* The dispatcher loop deliberately catches every backend failure, resolves
  all waiters of the failed batch with ``BackendError`` and continues, so no
  future is left hanging and the scheduler slot is always released in a
  ``finally`` block.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .errors import (
    BackendError,
    ClientDisconnected,
    GatewayError,
    RequestTimeout,
    ServiceUnavailable,
)

logger = logging.getLogger("embedding_gateway.batcher")


@dataclass
class PendingRequest:
    """A single gateway call waiting for (or riding along) a backend batch."""

    request_id: str
    texts: list[str]
    model: str
    deadline: float  # loop.time() monotonic deadline
    enqueued_at: float
    future: asyncio.Future[list[list[float]]] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    # pending -> dispatched -> finished | cancelled
    state: str = "pending"
    batch_id: str | None = None
    batch_size: int | None = None


@dataclass
class BatchResult:
    """Vectors for one caller plus the batch they were served from."""

    vectors: list[list[float]]
    batch_id: str
    batch_size: int


@dataclass
class BatchRecord:
    batch_id: str
    texts: int
    requests: int
    started_at: float
    finished_at: float | None = None
    outcome: str | None = None
    error: str | None = None


class BatchScheduler:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self.settings = settings
        self.client = client
        self._queue: deque[PendingRequest] = deque()
        self._wake = asyncio.Event()
        self._closing = False
        self._dispatcher: asyncio.Task[None] | None = None
        self._in_flight = False
        self._batch_counter = 0
        self._stats: dict[str, int] = {
            "accepted_requests": 0,
            "completed_requests": 0,
            "timed_out_requests": 0,
            "disconnected_requests": 0,
            "failed_requests": 0,
            "batches_dispatched": 0,
            "backend_errors": 0,
            "backend_texts_sent": 0,
        }
        self._last_batch: dict[str, Any] | None = None

    # ------------------------------------------------------------------ life
    async def start(self) -> None:
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(
                self._dispatch_loop(), name="embedding-batch-dispatcher"
            )

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        dispatcher = self._dispatcher
        self._dispatcher = None
        if dispatcher is not None:
            try:
                await dispatcher
            except asyncio.CancelledError:  # pragma: no cover - safety net
                pass
        # Anything still queued never got a chance to be sent.
        while self._queue:
            item = self._queue.popleft()
            if item.state == "pending":
                item.state = "cancelled"
            if not item.future.done():
                item.future.set_exception(ServiceUnavailable("gateway shutting down"))

    # --------------------------------------------------------------- submit
    async def submit(
        self,
        *,
        request_id: str,
        texts: list[str],
        model: str,
        deadline: float,
        request: Any | None = None,
    ) -> BatchResult:
        """Queue ``texts`` and wait until the batch result or a failure.

        ``deadline`` is an absolute ``loop.time()`` value computed from the
        moment the HTTP request was accepted.
        """
        loop = asyncio.get_running_loop()
        self._stats["accepted_requests"] += 1

        if self._closing:
            raise ServiceUnavailable("gateway is shutting down")

        remaining = deadline - loop.time()
        if remaining <= 0:
            self._stats["timed_out_requests"] += 1
            raise RequestTimeout("request deadline elapsed before queueing")

        item = PendingRequest(
            request_id=request_id,
            texts=texts,
            model=model,
            deadline=deadline,
            enqueued_at=loop.time(),
        )
        self._queue.append(item)
        self._wake.set()

        result_task = asyncio.ensure_future(item.future)
        disconnect_task: asyncio.Task[bool] | None = None
        if request is not None:
            disconnect_task = asyncio.create_task(
                self._watch_disconnect(request), name=f"disconnect-{request_id}"
            )

        waiters: set[asyncio.Future[Any]] = {result_task}
        if disconnect_task is not None:
            waiters.add(disconnect_task)

        try:
            while True:
                remaining = deadline - loop.time()
                done, _ = await asyncio.wait(
                    waiters,
                    timeout=max(0.0, remaining),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task is not None and disconnect_task in done:
                    if disconnect_task.result():
                        self._abandon(item, "disconnected")
                        self._stats["disconnected_requests"] += 1
                        raise ClientDisconnected(
                            "client disconnected before result"
                        )
                    # The disconnect probe gave up without observing a real
                    # disconnect; stop waiting on it.
                    waiters.discard(disconnect_task)
                    continue
                if result_task in done:
                    if result_task.cancelled():
                        # The dispatcher cancelled the future when it noticed
                        # the deadline had passed (an expired request is
                        # never sent).  Surface it as a timeout, not as an
                        # outer cancellation.
                        self._stats["timed_out_requests"] += 1
                        raise RequestTimeout(
                            "request deadline exceeded while waiting"
                        )
                    return BatchResult(
                        vectors=result_task.result(),
                        batch_id=item.batch_id or "batch-0",
                        batch_size=item.batch_size or len(texts),
                    )
                # Neither a result nor a confirmed disconnect: deadline hit.
                self._abandon(item, "timed out")
                self._stats["timed_out_requests"] += 1
                raise RequestTimeout("request deadline exceeded while waiting")
        except asyncio.CancelledError:
            # The ASGI server cancelled the handler (client vanished or
            # shutdown).  Remove the item if it is still pending; a dispatched
            # item is marked abandoned so its late vectors are dropped.
            was_dispatched = item.state == "dispatched"
            self._abandon(item, "cancelled")
            if was_dispatched:
                self._stats["disconnected_requests"] += 1
            raise
        finally:
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            if item.future.done() and not item.future.cancelled():
                # Mark a backend exception as retrieved so asyncio does not
                # log "exception was never retrieved" when the caller left
                # at the same moment the batch failed.
                item.future.exception()
            elif not item.future.done():
                item.future.cancel()

    async def _watch_disconnect(self, request: Any) -> bool:
        """Poll until the client is gone.

        Returns ``True`` for an observed disconnect and ``False`` when the
        server cannot observe disconnects at all, in which case :meth:`submit`
        stops waiting on this probe.
        """
        interval = self.settings.disconnect_poll_interval_ms / 1000.0
        while True:
            try:
                disconnected = await request.is_disconnected()
            except Exception:  # noqa: BLE001 - third-party ASGI implementations
                # This server cannot observe disconnects; stop probing rather
                # than reporting a false positive.  The deadline still bounds
                # how long the caller can wait.
                logger.debug("disconnect polling unavailable for this server")
                return False
            if disconnected:
                return True
            await asyncio.sleep(interval)

    def _abandon(self, item: PendingRequest, reason: str) -> None:
        """Best-effort, synchronous dequeue of a caller that gave up.

        No ``await`` happens in this method, so in a single-threaded event
        loop it is atomic with respect to the dispatcher's dequeue/mark logic.
        """
        if item.state == "pending":
            try:
                self._queue.remove(item)
            except ValueError:
                pass
            item.state = "cancelled"
            logger.debug("request %s abandoned while pending: %s", item.request_id, reason)
        elif item.state == "dispatched":
            # The batch is already on the wire.  Mark it so the dispatcher
            # discards the late vectors; it must still await the response.
            item.state = "cancelled"
            logger.debug("request %s abandoned while in flight: %s", item.request_id, reason)
        if not item.future.done():
            item.future.cancel()

    # ------------------------------------------------------------ dispatch
    async def _dispatch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closing:
            if not self._queue:
                self._wake.clear()
                if not self._queue:
                    await self._wake.wait()
                continue

            batch = await self._collect_batch(loop)
            if not batch:
                # Everything in the queue expired or was abandoned between
                # wake-up and collection; yield briefly instead of spinning.
                await asyncio.sleep(0.001)
                continue
            await self._send_batch(batch, loop)

    async def _collect_batch(
        self, loop: asyncio.AbstractEventLoop
    ) -> list[PendingRequest]:
        batch: list[PendingRequest] = []
        total = 0
        max_batch = self.settings.max_texts_per_batch
        window_end: float | None = None
        model: str | None = None

        while True:
            # Expired/abandoned entries are removed wherever they sit so they
            # are never sent and never hold a backend slot.
            while self._queue:
                head = self._queue[0]
                if head.state == "pending" and loop.time() < head.deadline:
                    break
                self._queue.popleft()
                if head.state == "pending":
                    head.state = "cancelled"
                    if not head.future.done():
                        head.future.cancel()

            if not self._queue:
                if not batch:
                    return []
                if total >= max_batch:
                    break
                # Queue drained before the batch is full.  If the merge window
                # opened by the batch's first request is still running, wait
                # for late arrivals; a backlogged batch (window already
                # elapsed while the backend was busy) ships immediately.
                remaining = (window_end or 0.0) - loop.time()
                if remaining <= 0:
                    break
                self._wake.clear()
                if self._queue:
                    continue
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                except TimeoutError:
                    pass
                continue

            head = self._queue[0]
            if batch:
                if head.model != model:
                    break
                if total + len(head.texts) > max_batch:
                    # A single request is never split; leave it for the next
                    # batch instead of exceeding the backend limit.
                    break
            else:
                # The window belongs to the first request actually collected.
                model = head.model
                window_end = head.enqueued_at + (
                    self.settings.batch_window_ms / 1000.0
                )

            self._queue.popleft()
            head.state = "dispatched"
            batch.append(head)
            total += len(head.texts)

            if total >= max_batch:
                break

        return batch

    async def _send_batch(
        self,
        batch: list[PendingRequest],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._batch_counter += 1
        batch_id = f"batch-{self._batch_counter}"
        # Drop anything that expired or disconnected between dequeue and this
        # point: such a request was never sent, so it must not consume backend
        # capacity or appear in the payload.
        now = loop.time()
        live: list[PendingRequest] = []
        for item in batch:
            if item.state != "dispatched":
                continue
            if now >= item.deadline:
                item.state = "cancelled"
                if not item.future.done():
                    item.future.cancel()
                continue
            live.append(item)
        batch = live
        if not batch:
            return
        flat_inputs = [text for item in batch for text in item.texts]
        model = batch[0].model
        record = BatchRecord(
            batch_id=batch_id,
            texts=len(flat_inputs),
            requests=len(batch),
            started_at=loop.time(),
        )
        self._in_flight = True
        self._stats["batches_dispatched"] += 1
        self._stats["backend_texts_sent"] += len(flat_inputs)

        try:
            vectors = await self._call_backend(model, flat_inputs)
            per_request = self._split_vectors(batch, vectors)
            for item, item_vectors in zip(batch, per_request):
                item.batch_id = batch_id
                item.batch_size = len(flat_inputs)
                if item.state == "dispatched" and not item.future.done():
                    item.state = "finished"
                    item.future.set_result(item_vectors)
                    self._stats["completed_requests"] += 1
                elif item.state == "dispatched":
                    item.state = "cancelled"
            record.outcome = "ok"
        except Exception as exc:  # noqa: BLE001 - must never kill dispatcher
            error = self._as_backend_error(exc)
            self._stats["backend_errors"] += 1
            record.outcome = "error"
            record.error = error.message
            for item in batch:
                if item.state == "dispatched":
                    item.state = "cancelled"
                    if not item.future.done():
                        item.future.set_exception(error)
                    self._stats["failed_requests"] += 1
        finally:
            self._in_flight = False
            record.finished_at = loop.time()
            self._last_batch = {
                "batch_id": record.batch_id,
                "texts": record.texts,
                "requests": record.requests,
                "duration_ms": round(
                    (record.finished_at - record.started_at) * 1000, 2
                ),
                "outcome": record.outcome,
                "error": record.error,
            }
            # Wake the loop in case submissions arrived during the call; the
            # dispatch loop would otherwise still be parked on the event.
            self._wake.set()

        # A caller may have timed out between the response arriving and the
        # future being resolved; that is handled by the state check above.

    async def _call_backend(
        self, model: str, inputs: list[str]
    ) -> list[dict[str, Any]]:
        headers = {"content-type": "application/json"}
        if self.settings.backend_api_key:
            headers["authorization"] = f"Bearer {self.settings.backend_api_key}"
        payload = {"model": model, "input": inputs}
        try:
            response = await self.client.post(
                self.settings.backend_url, json=payload, headers=headers
            )
        except httpx.TimeoutException as exc:
            raise BackendError(f"backend timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise BackendError(f"backend request failed: {exc}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            detail = response.text[:500].replace("\n", " ")
            raise BackendError(
                f"backend returned HTTP {response.status_code}: {detail}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise BackendError("backend returned invalid JSON") from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise BackendError("backend response is missing a 'data' list")
        return data

    def _split_vectors(
        self, batch: list[PendingRequest], data: list[dict[str, Any]]
    ) -> list[list[list[float]]]:
        total = sum(len(item.texts) for item in batch)
        if len(data) != total:
            raise BackendError(
                f"backend returned {len(data)} embeddings for {total} inputs"
            )

        ordered: list[list[float] | None] = [None] * total
        for position, entry in enumerate(data):
            if not isinstance(entry, dict):
                raise BackendError("backend embedding entries must be objects")
            index = entry.get("index", position)
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < total
            ):
                raise BackendError(f"backend returned invalid index {index!r}")
            if ordered[index] is not None:
                raise BackendError(f"backend returned duplicate index {index}")
            embedding = entry.get("embedding")
            if not isinstance(embedding, list) or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in embedding
            ):
                raise BackendError("backend embedding is not a list of numbers")
            ordered[index] = [float(value) for value in embedding]

        if any(vector is None for vector in ordered):
            raise BackendError("backend response is missing embedding indices")

        result: list[list[list[float]]] = []
        cursor = 0
        for item in batch:
            count = len(item.texts)
            chunk = ordered[cursor : cursor + count]
            # ``ordered`` was validated above, so no entry is None here.
            result.append([vector for vector in chunk if vector is not None])
            cursor += count
        return result

    @staticmethod
    def _as_backend_error(exc: Exception) -> BackendError:
        if isinstance(exc, BackendError):
            return exc
        if isinstance(exc, GatewayError):
            return BackendError(exc.message)
        return BackendError(f"backend batch failed: {exc}")

    # --------------------------------------------------------------- status
    def stats(self) -> dict[str, Any]:
        snapshot = dict(self._stats)
        snapshot["pending_requests"] = sum(
            1 for item in self._queue if item.state == "pending"
        )
        snapshot["in_flight"] = self._in_flight
        snapshot["last_batch"] = self._last_batch
        return snapshot
