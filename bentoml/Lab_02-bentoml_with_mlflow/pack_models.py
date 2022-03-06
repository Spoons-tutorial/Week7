from bentoml_api import ModelApi

import mlflow
from mlflow.tracking import MlflowClient

MODEL_STAGE = 'Production'
MODEL_NAME1 = 'sk-learn-rf-reg-mtcars-model'
MODEL_NAME2 = 'xgb-clf-wine-model'

mlflow.set_tracking_uri('http://localhost:5000')
client = MlflowClient()

MODEL_URI1 = f"models:/{MODEL_NAME1}/{MODEL_STAGE}"
MODEL_URI2 = f"models:/{MODEL_NAME2}/{MODEL_STAGE}"

rdf = mlflow.sklearn.load_model(MODEL_URI1)
xgb = mlflow.sklearn.load_model(MODEL_URI2)

bento_service = ModelApi()
bento_service.pack("mtcars_rf", rdf)
bento_service.pack("winequal_xgb", xgb)

saved_path = bento_service.save()