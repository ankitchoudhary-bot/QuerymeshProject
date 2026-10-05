"""Structured error responses.

Every `QueryMeshError` (Phase 23) maps to a JSON `ErrorResponse` with an
appropriate HTTP status — `retryable` on the error drives the status
class (503 for retryable/transient, 400/422/500 for not). A raw,
un-typed exception never leaks its message to the client — it's logged
server-side and returned as a generic message instead, so internal
details (stack traces, connection strings) never reach a caller.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from querymesh.api.dependencies import get_or_create_correlation_id, get_or_create_request_id
from querymesh.api.models import ErrorResponse
from querymesh.auth.errors import AuthenticationError, AuthorizationError
from querymesh.errors import ConfigurationError, QueryMeshError, ValidationError
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


def _error_response(
    request: Request, *, error: str, message: str, retryable: bool, status_code: int
) -> JSONResponse:
    request_id = get_or_create_request_id(request)
    correlation_id = get_or_create_correlation_id(request)
    body = ErrorResponse(
        error=error,
        message=message,
        retryable=retryable,
        request_id=request_id,
        correlation_id=correlation_id,
    )
    # Don't set X-Request-ID/X-Correlation-ID headers here: RequestIDMiddleware
    # (app.py) is the outermost ASGI layer and already appends both to every
    # response, error paths included — setting them again here produced a
    # real, observed bug: duplicate header values (e.g. "id, id"), since
    # ASGI/HTTP allow repeated header names and most clients join them.
    return JSONResponse(status_code=status_code, content=body.model_dump())


async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    return _error_response(
        request, error="ValidationError", message=str(exc), retryable=False, status_code=422
    )


async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
    response = _error_response(
        request, error="AuthenticationError", message=str(exc), retryable=False, status_code=401
    )
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


async def authorization_error_handler(request: Request, exc: AuthorizationError) -> JSONResponse:
    return _error_response(
        request, error="AuthorizationError", message=str(exc), retryable=False, status_code=403
    )


async def configuration_error_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
    logger.error("Configuration error handling request: %s", exc)
    return _error_response(
        request,
        error="ConfigurationError",
        message="The server is misconfigured. Please contact the operator.",
        retryable=False,
        status_code=500,
    )


async def query_mesh_error_handler(request: Request, exc: QueryMeshError) -> JSONResponse:
    logger.error("QueryMeshError handling request (retryable=%s): %s", exc.retryable, exc)
    status_code = 503 if exc.retryable else 500
    return _error_response(
        request,
        error=type(exc).__name__,
        message=str(exc),
        retryable=exc.retryable,
        status_code=status_code,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception handling request: %s", exc, exc_info=True)
    return _error_response(
        request,
        error="InternalError",
        message="An unexpected error occurred.",
        retryable=False,
        status_code=500,
    )


def timeout_error_response(request: Request, timeout_seconds: float) -> JSONResponse:
    """Not registered as an exception handler — called directly from the
    route on `asyncio.TimeoutError`, since a timed-out task is cancelled,
    not something to re-raise and let a handler catch.
    """
    return _error_response(
        request,
        error="RequestTimeout",
        message=f"The request did not complete within {timeout_seconds}s",
        retryable=True,
        status_code=504,
    )
