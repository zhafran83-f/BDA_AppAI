from fastapi import APIRouter, Request

from ...schemas.purchase import PurchaseRequest
from ...schemas.responses import PurchaseResponse
from ..dependencies import invoke_service

router = APIRouter(tags=["Purchase"])


@router.post("/purchase/predict", response_model=PurchaseResponse)
async def predict(payload: PurchaseRequest, request: Request):
    return await invoke_service(request, "purchase", "predict", payload.model_dump())
