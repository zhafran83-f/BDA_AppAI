"""Image decoding with no unverified resize or interpolation."""
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_FORMATS = frozenset({"JPEG", "PNG"})


def preprocess_image(data: bytes) -> np.ndarray:
    """Accept an encoded RGB JPEG/PNG already sized 224x224; return one batch."""
    if not isinstance(data, bytes) or not data:
        raise ValueError("Image must be non-empty encoded bytes.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 5 MiB upload limit.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ValueError("Only JPEG and PNG images are supported.")
            if image.size != IMAGE_SIZE:
                raise ValueError(
                    "Image must be exactly 224x224 pixels; resize is not enabled "
                    "until the training interpolation is verified."
                )
            if image.mode != "RGB":
                raise ValueError("Image must use RGB mode with three channels.")
            image.verify()
        # Reopen after verify(); never resize or silently convert color modes.
        with Image.open(BytesIO(data)) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError("Image must have normal EXIF orientation.")
            pixels = np.array(image, dtype=np.float32)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("Image cannot be decoded or is damaged.") from exc
    return np.expand_dims(pixels / np.float32(255.0), axis=0)
