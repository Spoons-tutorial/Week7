from fastapi import APIRouter, HTTPException
import numpy as np
import redis
import mlflow
import pandas as pd

from app.database import schema
from app.database.db import engine
from app.basemodels import TitanicInfo


schema.Base.metadata.create_all(bind=engine)
# app/database/schema.py에서 정의한 테이블이 없으면 생성합니다.

pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
client = redis.Redis(connection_pool=pool)

router = APIRouter(prefix="/titanic")

@router.post("/")
def predict_titanic(titanic_info:TitanicInfo):
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('titanic')
    
    # experiment이름으로 모든 runs를 찾습니다.
    exp = mlflow.get_experiment_by_name('titanic')
    exp_id = exp.experiment_id
    runs = mlflow.search_runs([exp_id])

    # metric에 대해 가장 나은 모델을 가져옵니다.
    best_score = runs['metrics.accuracy'].max()
    best_run = runs[runs['metrics.accuracy'] == best_score]
    best_run_id = best_run.run_id.values[0]

    titanic_target_names = {
        0: 'Dead',
        1: 'Survived'
    }
    test_x = pd.DataFrame({"Pclass": [2], "Sex": [0], "Fare": [300.0]})

    # 모델 uri를 바탕으로 모델을 load 해옵니다.
    best_model_uri = f'runs:/{best_run_id}/titanic_model'
    loaded_model = mlflow.sklearn.load_model(best_model_uri)

    result = loaded_model.predict(test_x)[0]

    return titanic_target_names[result]