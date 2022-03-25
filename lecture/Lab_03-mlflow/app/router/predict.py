import numpy as np
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from app.utils import load_rf_clf
from app.basemodels import IrisInfo

router = APIRouter(prefix="/iris")


@router.post("/predict")
def predict(iris_info: IrisInfo):
    model = None

    try:
        model = load_rf_clf("iris_rf_model")

        input_tensor = np.array([[*iris_info.dict().values()]])

        output_tensor = model.predict(input_tensor).tolist()
    except Exception as e:
        raise HTTPException(
            detail=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    result = {"result": output_tensor}

    return JSONResponse(content=result)
