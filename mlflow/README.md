# MLFlow Tutorial

## Env Setting

- `pip install -r requirements.txt`

## postgres docker

- `docker run --rm -P -p 127.0.0.1:5431:5432 -e POSTGRES_PASSWORD=0000 -e POSTGRES_USER=spoons --name pgtest postgres:13.4`

## mlflow dashboard

- `mlflow server --backend-store-uri postgresql://spoons:0000@localhost:5431/spoons --default-artifact-root $(pwd) -p 5001`

## workflow

- `python mlflow_tracking.py` 로 학습을 진행합니다.
- `python mlflow_inference.py` 로 가장 성능이 좋은 모델을 불러와 inference를 진행합니다.