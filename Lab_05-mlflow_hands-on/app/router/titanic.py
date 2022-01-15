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
    pass