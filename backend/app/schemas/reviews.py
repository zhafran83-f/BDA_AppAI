from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from ..preprocessing.reviews import MAX_REVIEW_CHARACTERS


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: Annotated[
        StrictStr, Field(min_length=1, max_length=MAX_REVIEW_CHARACTERS)
    ]
