"""Errors shared by the gateway request and batching layers."""

from __future__ import annotations


class GatewayError(Exception):
    """Base class for failures that can be mapped to an HTTP response."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class RequestTimeout(GatewayError):
    """The request deadline (measured from acceptance) expired."""

    status_code = 504
    code = "request_timeout"


class ClientDisconnected(GatewayError):
    """The HTTP client went away before its embedding was delivered."""

    status_code = 499
    code = "client_disconnected"


class BackendError(GatewayError):
    """The single-slot embedding backend failed or returned invalid data."""

    status_code = 502
    code = "backend_error"


class ServiceUnavailable(GatewayError):
    """The gateway is shutting down and cannot serve this request."""

    status_code = 503
    code = "service_unavailable"


class ValidationError(GatewayError):
    """The request is well-formed JSON but violates gateway limits."""

    status_code = 422
    code = "validation_error"
