from fastapi import APIRouter, Request

from ...schemas.responses import HealthResponse, ReadyResponse, ErrorResponse
from ..dependencies import get_runtime
from ..errors import APIError

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ErrorResponse}})
async def ready(request: Request):
    runtime = get_runtime(request)
    models = runtime.snapshot()
    if not all(state == "ready" for state in models.values()):
        raise APIError(
            503, "not_ready", "One or more AI models are not loaded or unavailable.",
            {"models": models},
        )
    return {"status": "ready", "models": models}
