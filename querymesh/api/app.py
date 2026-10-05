"""QueryMesh API app factory.

`create_app()` takes an optional pre-built `AppState` (or the pieces to
build one) so tests can inject a fake compiled graph — the same
dependency-injection pattern used throughout the rest of the project — 
without needing a real OpenSearch/LLM/Redis available. Production startup
(`main()`) calls it with no override, so the lifespan handler builds the
real `AppState` via `default_app_state()`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from querymesh.api.dependencies import (
    CORRELATION_ID_HEADER,
    REQUEST_ID_HEADER,
    AppState,
    _correlation_id_var,
    _request_id_var,
    default_app_state,
)
from querymesh.api.errors import (
    authentication_error_handler,
    authorization_error_handler,
    configuration_error_handler,
    query_mesh_error_handler,
    unhandled_exception_handler,
    validation_error_handler,
)
from querymesh.api.routes import router
from querymesh.auth.errors import AuthenticationError, AuthorizationError
from querymesh.errors import ConfigurationError, QueryMeshError, ValidationError
from querymesh.observability.logging import get_logger

logger = get_logger(__name__)


def _get_header(scope: dict, name: str) -> str | None:
    name_bytes = name.lower().encode("latin-1")
    for key, value in scope.get("headers") or []:
        if key == name_bytes:
            return value.decode("latin-1")
    return None


class RequestIDMiddleware:
    """Pure ASGI middleware (not `BaseHTTPMiddleware`) for two reasons:

    1. `BaseHTTPMiddleware` has a well-documented Starlette quirk where it
       re-raises an exception via `call_next()` even after a registered
       exception handler further in has already produced a response for
       it — which broke our own error handling (a route's exception was
       reaching the test client as a raw exception instead of the 500
       JSON response `unhandled_exception_handler` built). Pure ASGI
       middleware wraps `send` directly and has no such re-raising step.
    2. It sets `request_id`/`correlation_id` into contextvars *before*
       calling the inner app, so every dependency and every exception
       handler — however deep — sees the exact same IDs this middleware
       puts in the response headers. Previously, the middleware and the
       route each generated their own random UUID when the client didn't
       supply one, so the header and the response body disagreed.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _get_header(scope, REQUEST_ID_HEADER) or str(uuid.uuid4())
        correlation_id = _get_header(scope, CORRELATION_ID_HEADER) or request_id

        request_id_token = _request_id_var.set(request_id)
        correlation_id_token = _correlation_id_var.set(correlation_id)

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((REQUEST_ID_HEADER.lower().encode("latin-1"), request_id.encode("latin-1")))
                headers.append(
                    (CORRELATION_ID_HEADER.lower().encode("latin-1"), correlation_id.encode("latin-1"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            _request_id_var.reset(request_id_token)
            _correlation_id_var.reset(correlation_id_token)


def create_app(app_state: AppState | None = None) -> FastAPI:
    """Build the FastAPI app.

    `app_state`, when given, is used as-is (the test-injection path) and
    the lifespan handler is skipped entirely. When omitted, the lifespan
    handler builds the real `AppState` at startup via `default_app_state()`
    — the production path, which fails fast on missing configuration.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app_state is not None:
            app.state.querymesh = app_state
        else:
            logger.info("Initializing QueryMesh API application state...")
            app.state.querymesh = default_app_state()
            logger.info("QueryMesh API ready.")
        yield

    app = FastAPI(
        title="QueryMesh API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestIDMiddleware)
    app.include_router(router)

    app.add_exception_handler(ValidationError, validation_error_handler)
    app.add_exception_handler(AuthenticationError, authentication_error_handler)
    app.add_exception_handler(AuthorizationError, authorization_error_handler)
    app.add_exception_handler(ConfigurationError, configuration_error_handler)
    app.add_exception_handler(QueryMeshError, query_mesh_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    return app


def main() -> None:  # pragma: no cover - manual entrypoint, not unit tested
    import uvicorn

    from querymesh.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "querymesh.api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=settings.qm_env == "development",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
