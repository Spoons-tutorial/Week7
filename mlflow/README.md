# MLFlow

## Lab1

- mlflow docker를 띄울 때 [--backend-store-uri](https://www.mlflow.org/docs/latest/tracking.html#backend-stores)를 설정하여서 postgresdb에 실험정보가 기록되도록 합니다.
- [train.py](Lab_01-mlflow_train_inference/train.py)
  ```python
    # logging
    mlflow.log_param("train_dir", train_dir)
    mlflow.log_param("train dataset count", len(train_x))
    mlflow.log_param("target class count", len(set(train_y)))
    mlflow.log_artifact("train.csv")

    mlflow.log_metric(METRIC, score)
    mlflow.sklearn.log_model(model, MODEL_PATH)
  ```
  - `log_param`: 현재 실험의 파라미터를 기록합니다. + `log_params`를 통해 dictionary 정보를 기록할 수 있습니다.
  - `log_artifact`: 로컬의 file이나 directory를 기록합니다.
    - 예제에서는 실험에 사용된 학습데이터를 기록하였습니다.
  - `sklearn.log_model`: scikit-learn model을 기록합니다. `pytorch.log_model` 등 다양한 형태를 지원합니다.

## Lab2

- [train.py](Lab_02-mlflow_model_staging/train.py)
  ```python
    mlflow.sklearn.log_model(
        model,
        MODEL_PATH,
        registered_model_name='sk-learn-rf-clf-model'
    )
  ```
  - Lab1에 비해 달라진 점은 model logging시에 이름을 지정합니다.

- [register.py](Lab_02-mlflow_model_staging/register.py)
  ```python
    client = MlflowClient()
    client.transition_model_version_stage(
        name="sk-learn-rf-clf-model",
        version=MODEL_VERSION,
        stage=MODEL_STAGE1
    )
  ```
  - `name`과 `version`으로 모델을 찾아서 `stage`를 바꾸는 코드입니다.

- [inference.py](Lab_02-mlflow_model_staging/inference.py)
  ```python
    model_uri2 = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    loaded_model_by_stage = mlflow.sklearn.load_model(model_uri2)
  ```
  - inference시에도 model의 `name`과 `stage`를 기반으로 모델을 불러올 수 있습니다.
  - Way1을 참고하시면 `name`과 `version`을 기반으로 모델을 불러올 수 있음을 확인하실 수 있습니다.
## Lab3 hands-on

- Lab1에서는 inference결과를 print하였습니다.
- hands-on에서는 fastapi 결과로 리턴하는것을 실습합니다.
  - titanic예제의 경우 실습코드를 활용하면 쉽게 만들 수 있습니다.
  - 방식에는 여러 방식이 있으니 선호하시는 방식을 이용하여 저장된 model을 load하여 inference결과를 리턴하는것을 목표로 합니다.
- [MLFlow documentation](https://mlflow.org/docs/latest/index.html)을 읽어가며 본인의 모델을 서빙하는 코드를 작성해봅시다.