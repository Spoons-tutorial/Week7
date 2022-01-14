from typing import Optional
from pydantic import BaseModel, Field

class TitanicInfo(BaseModel):
    Pclass: int
    Sex: int
    Fare: float
