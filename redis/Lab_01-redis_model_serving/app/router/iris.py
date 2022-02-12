import datetime
import pickle

import joblib
import numpy as np
import redis
import onnxruntime as rt
from fastapi import APIRouter, HTTPException
from redisai import Client
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from app.basemodels import IrisInfo
from app.database import schema
from app.database.db import engine
from app.database.query import SELECT_MODEL
from app.utils import load_rf_clf


schema.Base.metadata.create_all(bind=engine)
# app/database/schema.py에서 정의한 테이블이 없으면 생성합니다.

con = Client(host='localhost', port=6379)

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

    try:
        onx_model = con.modelget(REDIS_KEY)
        onx_model = onx_model['blob']
    except redis.exceptions.ResponseError:            
        model = load_rf_clf(model_path)
        initial_type = [('float_input', FloatTensorType([1,4]))]
        onx_model = convert_sklearn(model, initial_types=initial_type).SerializeToString()
        con.modelset(REDIS_KEY, 'onnx', 'cpu', onx_model)
    ###

    ### onnx
    sess = rt.InferenceSession(onx_model)
    test = np.array([*iris_info.dict().values()]).reshape(1,-1).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    result = sess.run([label_name], {input_name: test})[0][0]
    ###

    return {
        "result": f'{model_name} 모델로 예측한 결과는 {iris_target_names[result]}입니다.'
    }


