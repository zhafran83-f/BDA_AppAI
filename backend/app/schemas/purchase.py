from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from ..preprocessing.shopper_features import MONTHS, VISITOR_TYPES

NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Rate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Month = Literal[*MONTHS]
VisitorType = Literal[*VISITOR_TYPES]


class PurchaseRequest(BaseModel):
    """Raw fields only; encoding remains exclusively in PurchaseService."""
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    Administrative: NonNegativeInt
    Administrative_Duration: NonNegativeFloat
    Informational: NonNegativeInt
    Informational_Duration: NonNegativeFloat
    ProductRelated: NonNegativeInt
    ProductRelated_Duration: NonNegativeFloat
    BounceRates: Rate
    ExitRates: Rate
    PageValues: NonNegativeFloat
    SpecialDay: Rate
    OperatingSystems: NonNegativeInt
    Browser: NonNegativeInt
    Region: NonNegativeInt
    TrafficType: NonNegativeInt
    Weekend: StrictBool
    Month: Month
    VisitorType: VisitorType
