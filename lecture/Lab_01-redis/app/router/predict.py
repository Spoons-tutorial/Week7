import numpy as np
from fastapi import APIRouter, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from app.utils import *
from app.basemodels import IrisInfo

router = APIRouter(prefix="")


@router.post("/v1/iris/predict/rai")
def predict(iris_info: IrisInfo):
    try:
        if not check_key(model_name="random_forest_rai"):
            set_model_rai("assets/random_forest.pkl", "random_forest_rai")

        input_tensor = np.array([[*iris_info.dict().values()]], dtype=np.float32)

        output_tensor = predict_redisai("random_forest_rai", input_tensor)
    except Exception as e:
        return HTTPException(
            detail=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    result = {"result": output_tensor}

    return JSONResponse(content=result)


@router.post("/v1/iris/predict")
def predict(iris_info: IrisInfo, background_tasks: BackgroundTasks):
    try:
        model = get_model_redis("random_forest")

        if model is None:
            model = load_rf_clf("assets/random_forest.pkl")
            background_tasks.add_task(set_model_redis, model, "random_forest")

        input_tensor = np.array([[*iris_info.dict().values()]], dtype=np.float32)

        output_tensor = model.predict(input_tensor).tolist()
    except Exception as e:
        return HTTPException(
            detail=str(e),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    result = {"result": output_tensor}

    return JSONResponse(content=result)
