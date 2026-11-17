from fastapi import APIRouter, Request

from ...schemas.reviews import ReviewRequest
from ...schemas.responses import ReviewResponse
from ..dependencies import invoke_service

router = APIRouter(tags=["Reviews"])


@router.post("/reviews/cluster", response_model=ReviewResponse)
async def cluster(payload: ReviewRequest, request: Request):
    return await invoke_service(request, "reviews", "cluster", payload.text)
