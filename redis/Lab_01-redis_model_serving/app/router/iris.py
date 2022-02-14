import datetime
import pickle

import joblib
import numpy as np
import redis
import redisai as rai
from fastapi import APIRouter, HTTPException
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from app.basemodels import IrisInfo
from app.database import schema
from app.database.db import engine
from app.database.query import SELECT_MODEL
from app.utils import load_rf_clf

schema.Base.metadata.create_all(bind=engine)
# app/database/schema.py에서 정의한 테이블이 없으면 생성합니다.

redisai_client = rai.Client(host="localhost", port=6379)

router = APIRouter(prefix="/iris")

@router.post("/")
def predict_iris(iris_info: IrisInfo, model_name:str = 'rf_clf_model_0106') -> str:
    """iris_info를 입력받아 model prediction 결과를 반환합니다.

    Args:
        iris_info (IrisInfo): sepal_length, sepal_width, petal_length, petal_width 정보

    Returns:
        str: model prediction결과가 포함된 문자열
    """
    iris_target_names = {
        0: "setosa",
        1: "versicolor",
        2: "virginica"
    }
    try:
        with engine.connect() as conn:
            result = conn.execute(SELECT_MODEL.format(model_name)).fetchone()
            model_name, model_path = result
    except:
        raise HTTPException(status_code=404, detail="DB에 model_name이 존재하지 않습니다.")
    
    ### redis
    REDIS_KEY = 'redis_caching_model'

    # 데이터를 redisai에서 활용할 수 있도록 set하여 줍니다.
    test = np.array([[*iris_info.dict().values()]], dtype=np.float32)
    redisai_client.tensorset('input_tensor', test)

    # 모델이 존재하는지 체크합니다.
    is_model_exist = redisai_client.exists(REDIS_KEY)

    # 모델이 존재하지 않는 경우
    if not is_model_exist:
        loaded_model = load_rf_clf(model_path)

        initial_type = [("input_tensor", FloatTensorType([None, 4]))]

        onx_model = convert_sklearn(loaded_model, initial_types=initial_type)

        redisai_client.modelstore(
            key=REDIS_KEY, 
            backend='onnx', 
            device='cpu', 
            data=onx_model.SerializeToString()
        )

    # redisai modelrun method로 inference를 수행합니다.
    redisai_client.modelrun(key=REDIS_KEY,
                        inputs=['input_tensor'],
                        outputs=['output_tensor_class', 'output_tensor_prob'])
    
    result = int(redisai_client.tensorget('output_tensor_class'))

    return {
        "result": f'{model_name} 모델로 예측한 결과는 {iris_target_names[result]}입니다.'
    }


