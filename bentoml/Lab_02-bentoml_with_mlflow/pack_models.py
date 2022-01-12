from bentoml_api import ModelApi

import mlflow
from mlflow.tracking import MlflowClient

mt_exp_name = "mtcars"
mt_art_path = "mtcars_model"
wq_exp_name = "wine_quality"
wq_art_path = "winequal_model"


mlflow.set_tracking_uri('http://localhost:8000')
client = MlflowClient()

mtcars_exp_id = client.get_experiment_by_name(mt_exp_name).experiment_id
mtcars_run_id = mlflow.search_runs([mtcars_exp_id])['run_id'][0]

rdf = mlflow.sklearn.load_model(f"runs:/{mtcars_run_id}/{mt_art_path}")


winequal_exp_id = client.get_experiment_by_name(wq_exp_name).experiment_id
winequal_run_id = mlflow.search_runs([winequal_exp_id])['run_id'][0]

xgb = mlflow.xgboost.load_model(f"runs:/{winequal_run_id}/{wq_art_path}")

bento_service = ModelApi()
bento_service.pack("mtcars_rf", rdf)
bento_service.pack("winequal_xgb", xgb)

saved_path = bento_service.save()