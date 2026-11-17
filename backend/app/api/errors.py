"""One public error envelope; internal exceptions never enter HTTP responses."""
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from ..schemas.responses import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details=None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def error_response(status_code, code, message, details=None, headers=None):
    body = ErrorResponse(error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(
        status_code=status_code, content=body.model_dump(mode="json"), headers=headers
    )


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError):
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def request_error(request: Request, exc: RequestValidationError):
        # Deliberately omit input and ctx: these may contain sensitive request data.
        details = [
            {"location": list(issue["loc"]), "message": issue["msg"], "type": issue["type"]}
            for issue in exc.errors()
        ]
        return error_response(422, "validation_error", "Request validation failed.", details)

    @app.exception_handler(ResponseValidationError)
    async def response_error(request: Request, exc: ResponseValidationError):
        runtime = getattr(request.app.state, "runtime", None)
        module = getattr(request.state, "inference_module", None)
        if runtime is not None and module is not None:
            runtime.mark_unavailable(module)
        logger.error("Inference response did not match its public contract.")
        return error_response(500, "inference_error", "The inference result could not be produced.")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if exc.status_code == 400:
            return error_response(422, "validation_error", "Request body could not be parsed.")
        # Avoid forwarding exception detail from an arbitrary internal component.
        messages = {404: "Resource not found.", 405: "Method not allowed."}
        return error_response(
            exc.status_code, "http_error",
            messages.get(exc.status_code, "Request could not be processed."),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("Unhandled API exception.")
        return error_response(500, "internal_error", "An internal error occurred.")
