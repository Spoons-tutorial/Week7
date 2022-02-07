import mlflow
from mlflow.tracking import MlflowClient

if __name__ == '__main__':

    mlflow.set_tracking_uri('http://localhost:5000')

    # 현재 버전
    MODEL_VERSION = 1

    # 원하는 Stage값
    MODEL_STAGE1 = 'Production'
    MODEL_STAGE2 = 'Staging'

    client = MlflowClient()
    client.transition_model_version_stage(
        name="sk-learn-rf-clf-model",
        version=MODEL_VERSION,
        stage=MODEL_STAGE1
    )