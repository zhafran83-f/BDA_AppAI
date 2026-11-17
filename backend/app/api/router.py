from fastapi import APIRouter

from ..schemas.responses import ErrorResponse
from .routes import fashion, health, purchase, reviews

router = APIRouter()
router.include_router(health.router)
inference_router = APIRouter(
    prefix="/api/v1",
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
inference_router.include_router(fashion.router)
inference_router.include_router(reviews.router)
inference_router.include_router(purchase.router)
router.include_router(inference_router)
