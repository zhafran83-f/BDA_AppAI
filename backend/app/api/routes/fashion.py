from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile

from ...preprocessing.images import MAX_IMAGE_BYTES
from ...schemas.responses import FashionResponse, ErrorResponse
from ..dependencies import invoke_service
from ..errors import APIError

router = APIRouter(tags=["Fashion"])


@router.post(
    "/fashion/classify", response_model=FashionResponse,
    responses={413: {"model": ErrorResponse}},
)
async def classify(
    request: Request,
    image: Annotated[UploadFile, File(description="One RGB JPEG/PNG, exactly 224x224, at most 5 MiB.")],
):
    try:
        form = await request.form()
        items = list(form.multi_items())
        if len(items) != 1 or items[0][0] != "image":
            raise APIError(422, "validation_error", "Only one multipart field named image is allowed.")
        if image.size is not None and image.size > MAX_IMAGE_BYTES:
            raise APIError(413, "file_too_large", "Image exceeds the 5 MiB upload limit.")
        if (image.content_type or "").lower() not in {"image/png", "image/jpeg"}:
            raise APIError(422, "validation_error", "Image must be uploaded as JPEG or PNG.")
        content = await image.read(MAX_IMAGE_BYTES + 1)
        if len(content) > MAX_IMAGE_BYTES:
            raise APIError(413, "file_too_large", "Image exceeds the 5 MiB upload limit.")
        return await invoke_service(request, "fashion", "classify", content)
    finally:
        await image.close()
