import numpy as np
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException

from app.utils import load_rf_clf
from app.basemodels import IrisInfo

router = APIRouter(prefix='/iris')

@router.post('/predict')
def predict(iris_info: IrisInfo):
    model = None
    try:
        model = load_rf_clf('iris_rf_model')
        
    except Exception as e:
        raise HTTPException(detail=f"model is not loaded: {str(e)}", 
              status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    input_data = np.array([[*iris_info.dict().values()]])
    result = {'result': model.predict(input_data).tolist()}

    return JSONResponse(content=result)