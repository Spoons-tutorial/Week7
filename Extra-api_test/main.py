import joblib

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class MtcarInfo(BaseModel):
    cyl: float
    disp: float
    hp: int
    drat: float
    wt: float
    qsec: float
    vs: int
    am: int
    gear: int
    carb: int


    class Config: # swagger에서 보여줄 각 파라미터에 대한 예시를 설정할 수 있습니다.
        schema_extra={
            "example": {
                'cyl': 6,
                'disp': 160.0,
                'hp': 110,
                'drat': 3.90,
                'wt': 2.620,
                'qsec': 16.46,
                'vs': 0,
                'am': 1,
                'gear': 4,
                'carb': 4
            }
        }

@app.post("/")
def root(mt_car_info:MtcarInfo):
    model = joblib.load('./rf_reg')
    test = np.array([*mt_car_info.dict().values()]).reshape(1,-1)
    result = model.predict(test)[0]

    return result