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

## Lab2 hands-on

- Lab1에서는 inference결과를 print하였습니다.
- hands-on에서는 fastapi 결과로 리턴하는것을 실습합니다.
  - titanic예제의 경우 코드를 거의 그대로 활용하면 만들 수 있습니다.
- [MLFlow documentation](https://mlflow.org/docs/latest/index.html)을 읽어가며 본인의 모델을 서빙하는 코드를 작성해봅시다.