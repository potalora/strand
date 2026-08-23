from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute

INVALID_AUTH_REQUEST_BODY = {"detail": "Invalid authentication request."}
_CONTENT_FREE_ROUTE_NAMES = frozenset({"auth_register", "auth_login"})


class ContentFreeAuthValidationRoute(APIRoute):
    """Sanitize register/login validation without changing other route errors."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()
        if self.name not in _CONTENT_FREE_ROUTE_NAMES:
            return original

        async def content_free_handler(request: Request) -> Response:
            try:
                return await original(request)
            except RequestValidationError:
                return JSONResponse(status_code=422, content=INVALID_AUTH_REQUEST_BODY)

        return content_free_handler
