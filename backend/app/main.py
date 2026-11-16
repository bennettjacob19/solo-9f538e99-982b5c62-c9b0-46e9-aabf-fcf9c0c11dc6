"""A controllable stand-in for the single-slot embedding backend.

Endpoints
---------
POST /v1/embeddings   OpenAI-compatible embeddings endpoint.
GET  /healthz         Liveness probe.
GET  /admin/stats     Actual batching/concurrency counters and request log.
POST /admin/config    Change delay/error behaviour at runtime.
POST /admin/reset     Clear counters and logs (not the config).
GET  /docs            Interactive docs, including the admin endpoints.

The backend records how many requests it is actually serving at once.  Since
the gateway is supposed to be single-flight against this backend, any value
above 1 (and any batch larger than the configured limit) is flagged as a
violation and logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("mock_embedding_backend")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw == "" else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class BackendConfig(BaseModel):
    delay_ms: int = Field(default=0, ge=0, le=60_000)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    error_status: int = Field(default=503, ge=400, le=599)
    mode: Literal[
        "ok", "error", "malformed", "short", "duplicate_index", "shuffle"
    ] = "ok"
    shuffle: bool = False
    max_inputs: int = Field(default=8, ge=1, le=1024)


class BackendConfigPatch(BaseModel):
    delay_ms: int | None = Field(default=None, ge=0, le=60_000)
    error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    error_status: int | None = Field(default=None, ge=400, le=599)
    mode: Literal[
        "ok", "error", "malformed", "short", "duplicate_index", "shuffle"
    ] | None = None
    shuffle: bool | None = None
    max_inputs: int | None = Field(default=None, ge=1, le=1024)


def make_embedding(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dim:
        for byte in digest:
            if len(values) == dim:
                break
            values.append(round((byte / 127.5) - 1.0, 6))
        digest = hashlib.sha256(digest).digest()
    return values


class BackendState:
    def __init__(self) -> None:
        self.config = BackendConfig(
            delay_ms=_env_int("MOCK_DELAY_MS", 0),
            error_rate=_env_float("MOCK_ERROR_RATE", 0.0),
            error_status=_env_int("MOCK_ERROR_STATUS", 503),
            mode=os.getenv("MOCK_MODE", "ok"),  # type: ignore[arg-type]
            shuffle=_env_bool("MOCK_SHUFFLE_RESPONSE", False),
            max_inputs=_env_int("MOCK_MAX_INPUTS", 8),
        )
        self.dim = _env_int("MOCK_DIM", 8)
        self.current = 0
        self.max_concurrency = 0
        self.concurrency_violations = 0
        self.oversized_batches = 0
        self.total_requests = 0
        self.total_inputs = 0
        self.max_batch_seen = 0
        self.requests: list[dict[str, Any]] = []

    @property
    def in_flight(self) -> bool:
        return self.current > 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.model_dump(),
            "dim": self.dim,
            "current_concurrency": self.current,
            "max_concurrency": self.max_concurrency,
            "concurrency_violations": self.concurrency_violations,
            "oversized_batches": self.oversized_batches,
            "total_requests": self.total_requests,
            "total_inputs": self.total_inputs,
            "max_batch_seen": self.max_batch_seen,
            "requests": self.requests[-200:],
        }

    def reset(self) -> None:
        self.current = 0
        self.max_concurrency = 0
        self.concurrency_violations = 0
        self.oversized_batches = 0
        self.total_requests = 0
        self.total_inputs = 0
        self.max_batch_seen = 0
        self.requests = []


state = BackendState()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "mock_backend_error"}},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "mock backend ready: delay_ms=%s mode=%s dim=%s",
        state.config.delay_ms,
        state.config.mode,
        state.dim,
    )
    yield


app = FastAPI(
    title="Mock Embedding Backend",
    version="1.0.0",
    description=(
        "Single-slot local embedding backend with configurable latency and "
        "fault injection. It records actual concurrency so gateway batching "
        "can be verified."
    ),
    lifespan=lifespan,
)


@app.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/admin/stats", tags=["admin"])
async def admin_stats() -> dict[str, Any]:
    return state.snapshot()


@app.post("/admin/config", tags=["admin"])
async def admin_config(patch: BackendConfigPatch) -> dict[str, Any]:
    updates = patch.model_dump(exclude_none=True)
    state.config = state.config.model_copy(update=updates)
    logger.info("config updated: %s", updates)
    return state.config.model_dump()


@app.post("/admin/reset", tags=["admin"])
async def admin_reset() -> dict[str, Any]:
    state.reset()
    return state.snapshot()


@app.post(
    "/v1/embeddings",
    tags=["embeddings"],
    summary="Return deterministic embeddings for the supplied texts",
)
async def embeddings(request: Request) -> Any:
    started = time.monotonic()
    try:
        body = await request.json()
    except ValueError:
        return _error(400, "request body must be valid JSON")

    raw_input = body.get("input") if isinstance(body, dict) else None
    if isinstance(raw_input, str):
        texts = [raw_input]
    elif isinstance(raw_input, list) and all(isinstance(t, str) for t in raw_input):
        texts = raw_input
    else:
        return _error(400, "input must be a string or a list of strings")
    if not texts:
        return _error(400, "input must not be empty")

    model = body.get("model") or "local-demo"

    # Concurrency accounting (single-threaded event loop => atomic updates).
    state.current += 1
    state.max_concurrency = max(state.max_concurrency, state.current)
    concurrency_at_entry = state.current
    state.total_requests += 1
    state.total_inputs += len(texts)
    state.max_batch_seen = max(state.max_batch_seen, len(texts))
    if state.current > 1:
        state.concurrency_violations += 1
        logger.warning(
            "CONCURRENCY VIOLATION: %s simultaneous requests", state.current
        )
    if len(texts) > state.config.max_inputs:
        state.oversized_batches += 1
        logger.warning(
            "OVERSIZED BATCH: %s inputs (limit %s)",
            len(texts),
            state.config.max_inputs,
        )

    record: dict[str, Any] = {
        "started_at": started,
        "texts": texts,
        "batch_size": len(texts),
        "model": model,
        "concurrency_at_entry": concurrency_at_entry,
        "status": None,
        "ended_at": None,
    }
    state.requests.append(record)

    try:
        if state.config.delay_ms:
            await asyncio.sleep(state.config.delay_ms / 1000.0)

        if len(texts) > state.config.max_inputs:
            record["status"] = 400
            return _error(
                400,
                f"batch contains {len(texts)} inputs, limit is "
                f"{state.config.max_inputs}",
            )

        # error mode/rate are evaluated after the delay so the gateway must
        # wait for a slow failure just like a slow success.
        if state.config.mode == "error" or (
            state.config.error_rate > 0 and _roll(state.config.error_rate)
        ):
            record["status"] = state.config.error_status
            return _error(
                state.config.error_status,
                "injected backend failure",
            )

        if state.config.mode == "malformed":
            record["status"] = 200
            return {"model": model, "usage": {"prompt_tokens": 0}}

        if state.config.mode == "short" and len(texts) > 1:
            texts_for_response = texts[:-1]
        else:
            texts_for_response = texts

        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": make_embedding(text, state.dim),
            }
            for index, text in enumerate(texts_for_response)
        ]

        if state.config.mode == "duplicate_index" and len(data) > 1:
            data[1]["index"] = 0

        if state.config.mode == "shuffle" or state.config.shuffle:
            data.reverse()

        record["status"] = 200
        return {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": sum(len(t) for t in texts)},
        }
    except Exception as exc:
        record["status"] = 500
        logger.exception("backend error")
        return _error(500, f"internal backend error: {exc}")
    finally:
        record["ended_at"] = time.monotonic()
        record["duration_ms"] = round(
            (record["ended_at"] - started) * 1000, 2
        )
        state.current -= 1
        logger.info(
            "batch size=%s concurrency=%s status=%s duration_ms=%s texts=%s",
            len(texts),
            concurrency_at_entry,
            record["status"],
            record["duration_ms"],
            [t[:40] for t in texts],
        )


def _roll(rate: float) -> bool:
    import random

    return random.random() < rate


@app.get("/admin/log", tags=["admin"])
async def admin_log(limit: int = Query(default=50, ge=1, le=200)) -> list[dict]:
    return state.requests[-limit:]
