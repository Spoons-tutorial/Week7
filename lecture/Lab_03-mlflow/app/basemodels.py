from pydantic import BaseModel, Field


class IrisInfo(BaseModel):
    sepal_length: float = Field(..., ge=4.3, le=7.9)
    sepal_width: float = Field(..., ge=2.0, le=4.4)
    petal_length: float = Field(..., ge=1.0, le=6.9)
    petal_width: float = Field(..., ge=0.1, le=2.5)
