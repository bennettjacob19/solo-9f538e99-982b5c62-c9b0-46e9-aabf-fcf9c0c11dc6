"""Runtime configuration for the embedding batch gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    """Gateway settings.

    The defaults encode the product requirements: a request carries 1..4
    texts, a backend batch is at most 8 texts, and the first pending request
    is held for at most 20 ms while the backend is idle.
    """

    backend_url: str = "http://127.0.0.1:9000/v1/embeddings"
    backend_model: str = "local-demo"
    backend_api_key: str | None = None
    backend_timeout_seconds: float = 30.0
    backend_connect_timeout_seconds: float = 5.0

    batch_window_ms: int = 20
    max_texts_per_request: int = 4
    max_texts_per_batch: int = 8
    default_timeout_ms: int = 5000
    max_timeout_ms: int = 600_000
    disconnect_poll_interval_ms: int = 20

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            backend_url=os.getenv(
                "EMBEDDING_BACKEND_URL", cls.backend_url
            ),
            backend_model=os.getenv("EMBEDDING_MODEL", cls.backend_model),
            backend_api_key=os.getenv("EMBEDDING_API_KEY") or None,
            backend_timeout_seconds=_env_float(
                "EMBEDDING_BACKEND_TIMEOUT_SECONDS",
                cls.backend_timeout_seconds,
            ),
            backend_connect_timeout_seconds=_env_float(
                "EMBEDDING_BACKEND_CONNECT_TIMEOUT_SECONDS",
                cls.backend_connect_timeout_seconds,
            ),
            batch_window_ms=_env_int(
                "GATEWAY_BATCH_WINDOW_MS", cls.batch_window_ms
            ),
            max_texts_per_request=_env_int(
                "GATEWAY_MAX_TEXTS_PER_REQUEST",
                cls.max_texts_per_request,
            ),
            max_texts_per_batch=_env_int(
                "GATEWAY_MAX_TEXTS_PER_BATCH", cls.max_texts_per_batch
            ),
            default_timeout_ms=_env_int(
                "GATEWAY_DEFAULT_TIMEOUT_MS", cls.default_timeout_ms
            ),
            max_timeout_ms=_env_int(
                "GATEWAY_MAX_TIMEOUT_MS", cls.max_timeout_ms
            ),
            disconnect_poll_interval_ms=_env_int(
                "GATEWAY_DISCONNECT_POLL_INTERVAL_MS",
                cls.disconnect_poll_interval_ms,
            ),
        )

    def validate(self) -> None:
        if self.max_texts_per_request < 1:
            raise ValueError("max_texts_per_request must be >= 1")
        if self.max_texts_per_batch < self.max_texts_per_request:
            raise ValueError(
                "max_texts_per_batch must be >= max_texts_per_request"
            )
        if self.batch_window_ms < 0:
            raise ValueError("batch_window_ms must be >= 0")
        if self.backend_timeout_seconds <= 0:
            raise ValueError("backend timeout must be > 0")
        if self.disconnect_poll_interval_ms <= 0:
            raise ValueError("disconnect poll interval must be > 0")
        if self.default_timeout_ms < 1 or self.max_timeout_ms < 1:
            raise ValueError("timeouts must be >= 1 ms")
        if self.default_timeout_ms > self.max_timeout_ms:
            raise ValueError("default_timeout_ms exceeds max_timeout_ms")
