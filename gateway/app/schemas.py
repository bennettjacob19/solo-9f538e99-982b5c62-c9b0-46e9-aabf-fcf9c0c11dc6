"""API schemas for the gateway."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EmbeddingRequest(BaseModel):
    """One gateway request carrying 1..4 short texts.

    ``input`` follows the OpenAI embeddings shape (a string or a list of
    strings). ``texts`` is accepted as a convenience alias.
    """

    input: str | list[str] = Field(
        ...,
        description="One to four short texts (a lone string is also allowed).",
        examples=[["hello world", "second text"]],
    )
    timeout_ms: int | None = Field(
        default=None,
        ge=1,
        description=(
            "End-to-end deadline in milliseconds, starting when the gateway "
            "accepts the request. It covers queueing plus the backend call."
        ),
        examples=[1000],
    )
    model: str | None = Field(
        default=None,
        description="Optional model hint forwarded to the backend.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_texts_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "input" not in data and "texts" in data:
                data = {**data, "input": data["texts"]}
            if "input" not in data and "prompt" in data:
                data = {**data, "input": data["prompt"]}
        return data

    @field_validator("input")
    @classmethod
    def _normalize_input(cls, value: str | list[str]) -> str | list[str]:
        if isinstance(value, str):
            return value
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError("input must be a string or a list of strings")
        return value

    def as_texts(self) -> list[str]:
        return [self.input] if isinstance(self.input, str) else list(self.input)


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingItem]
    model: str
    batch_id: str = Field(description="Identifier shared by one backend batch.")
    batch_size: int = Field(
        description="Total number of texts sent to the backend in this batch."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class StatsResponse(BaseModel):
    accepted_requests: int
    completed_requests: int
    timed_out_requests: int
    disconnected_requests: int
    failed_requests: int
    batches_dispatched: int
    backend_errors: int
    backend_texts_sent: int
    pending_requests: int
    in_flight: bool
    last_batch: dict[str, Any] | None = None
