"""Test fixtures: a controllable in-process fake embedding backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from gateway.app.config import Settings
from gateway.app.main import create_app


def vector_for(text: str, dim: int = 4) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [round(digest[i] / 255.0, 6) for i in range(dim)]


@dataclass
class Call:
    texts: list[str]
    model: str
    start: float
    end: float | None = None
    status: int | None = None


@dataclass
class FakeBackend:
    """MockTransport handler that mimics a single-slot embedding backend."""

    delay: float = 0.0
    status: int = 200
    mode: str = "ok"  # ok | malformed | short | duplicate_index | bad_coalesce
    shuffle: bool = True
    dim: int = 4
    calls: list[Call] = field(default_factory=list)
    active: int = 0
    max_active: int = 0
    on_call: Any = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        loop = asyncio.get_running_loop()
        body = json.loads(request.content)
        texts = body["input"]
        call = Call(texts=list(texts), model=body.get("model", ""), start=loop.time())
        self.calls.append(call)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.on_call is not None:
            await self.on_call(len(self.calls))
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            call.status = self.status
            if self.status != 200:
                return httpx.Response(
                    self.status, json={"error": {"message": "injected failure"}}
                )
            if self.mode == "malformed":
                return httpx.Response(200, json={"object": "list"})
            if self.mode == "bad_coalesce":
                return httpx.Response(200, text="<html>not json</html>")
            if self.mode == "short":
                emitted = texts[:-1]
            else:
                emitted = texts
            data = [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": vector_for(text, self.dim),
                }
                for index, text in enumerate(emitted)
            ]
            if self.mode == "duplicate_index" and len(data) > 1:
                data[1]["index"] = 0
            if self.shuffle:
                data.reverse()
            return httpx.Response(
                200, json={"object": "list", "data": data, "model": "fake"}
            )
        finally:
            call.end = loop.time()
            self.active -= 1

    @property
    def batch_sizes(self) -> list[int]:
        return [len(call.texts) for call in self.calls]

    @property
    def sent_texts(self) -> list[str]:
        return [text for call in self.calls for text in call.texts]


def make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "backend_url": "http://fake.test/v1/embeddings",
        "backend_model": "fake-model",
        "backend_timeout_seconds": 10.0,
        "backend_connect_timeout_seconds": 2.0,
        "batch_window_ms": 20,
        "max_texts_per_request": 4,
        "max_texts_per_batch": 8,
        "default_timeout_ms": 5000,
        "max_timeout_ms": 600000,
        "disconnect_poll_interval_ms": 10,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@asynccontextmanager
async def gateway_client(
    backend: FakeBackend, **settings_overrides: Any
) -> AsyncIterator[tuple[Any, httpx.AsyncClient]]:
    settings = make_settings(**settings_overrides)
    app = create_app(
        settings, backend_transport=httpx.MockTransport(backend)
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway"
        ) as client:
            yield app, client


async def post(
    client: httpx.AsyncClient, texts: list[str], timeout_ms: int = 5000, **extra: Any
) -> httpx.Response:
    payload = {"input": texts, "timeout_ms": timeout_ms, **extra}
    return await client.post("/v1/embeddings", json=payload)
