"""FastAPI entrypoint for the embedding batch gateway."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .batcher import BatchScheduler
from .config import Settings
from .errors import GatewayError, ValidationError
from .schemas import (
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    HealthResponse,
    StatsResponse,
)

logger = logging.getLogger("embedding_gateway")


class AcceptanceTimeMiddleware:
    """Stamp the receipt time before the body is parsed.

    The per-request deadline must start when the gateway accepts the call,
    not when the request reaches the scheduler.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            state = scope.setdefault("state", {})
            state.setdefault("accepted_at", asyncio.get_running_loop().time())
        await self.app(scope, receive, send)


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def create_app(
    settings: Settings | None = None,
    *,
    backend_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = httpx.Timeout(
            settings.backend_timeout_seconds,
            connect=settings.backend_connect_timeout_seconds,
        )
        client = httpx.AsyncClient(
            timeout=timeout,
            transport=backend_transport,
            limits=httpx.Limits(
                max_connections=1, max_keepalive_connections=1
            ),
        )
        scheduler = BatchScheduler(settings, client)
        app.state.client = client
        app.state.scheduler = scheduler
        await scheduler.start()
        logger.info(
            "gateway ready: backend=%s window_ms=%s batch_max=%s",
            settings.backend_url,
            settings.batch_window_ms,
            settings.max_texts_per_batch,
        )
        try:
            yield
        finally:
            await scheduler.close()
            await client.aclose()

    app = FastAPI(
        title="Embedding Batch Gateway",
        version="1.0.0",
        description=(
            "Accepts 1-4 short texts per call, merges concurrent calls into "
            "single-flight backend batches of at most 8 texts, and enforces a "
            "deadline measured from request acceptance."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(AcceptanceTimeMiddleware)
    app.state.settings = settings

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(
        request: Request, exc: GatewayError
    ) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        first = errors[0] if errors else {}
        location = ".".join(str(part) for part in first.get("loc", ()))
        message = f"{location}: {first.get('msg', 'invalid request')}"
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": message.lstrip(": "),
                    "details": errors,
                }
            },
        )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        return HealthResponse()

    @app.get("/stats", response_model=StatsResponse, tags=["ops"])
    async def stats(request: Request) -> StatsResponse:
        return StatsResponse(**request.app.state.scheduler.stats())

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingResponse,
        tags=["embeddings"],
        summary="Embed 1-4 texts",
    )
    async def embeddings(
        payload: EmbeddingRequest, request: Request
    ) -> EmbeddingResponse:
        texts = payload.as_texts()
        if not 1 <= len(texts) <= settings.max_texts_per_request:
            raise ValidationError(
                f"input must contain between 1 and "
                f"{settings.max_texts_per_request} texts"
            )

        timeout_ms = (
            payload.timeout_ms
            if payload.timeout_ms is not None
            else settings.default_timeout_ms
        )
        if timeout_ms > settings.max_timeout_ms:
            raise ValidationError(
                f"timeout_ms must be <= {settings.max_timeout_ms}"
            )

        accepted_at = request.state.accepted_at
        deadline = accepted_at + timeout_ms / 1000.0
        request_id = uuid.uuid4().hex
        model = payload.model or settings.backend_model

        result = await request.app.state.scheduler.submit(
            request_id=request_id,
            texts=texts,
            model=model,
            deadline=deadline,
            request=request,
        )
        return EmbeddingResponse(
            data=[
                EmbeddingItem(index=index, embedding=vector)
                for index, vector in enumerate(result.vectors)
            ],
            model=model,
            batch_id=result.batch_id,
            batch_size=result.batch_size,
        )

    return app


app = create_app()
