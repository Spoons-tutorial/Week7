import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.tracking import MlflowClient

basedir = os.getenv("HOME")

exp_name = "mtcars"
artifact_path = "mtcars_model"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(exp_name)

client = MlflowClient()
exp_id = client.get_experiment_by_name(exp_name).experiment_id

with mlflow.start_run(experiment_id = exp_id):    
    data = pd.read_csv(f"{basedir}/Week7/bentoml/data/mtcars.csv")
    data = data.drop(labels = "Unnamed: 0", axis = 1)
    
    y = "mpg"
    X_data = data.loc[:, data.keys() != y]
    y_data = data[y]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

    params = {"n_estimators": 15, "min_samples_split": 3}
    rdf = RandomForestRegressor(**params,
                                criterion="squared_error",
                                random_state = 2022)
    rdf.fit(X_data.values, y_data.values)

    result = rdf.score(X_test.values, y_test.values)

    mlflow.log_params(params)
    mlflow.log_metric(key = "MSE", value = result)

    MODEL_VERSION = 1
    MODEL_STAGE = 'Production'
    MODEL_NAME = 'sk-learn-rf-reg-mtcars-model'

    # log model
    mlflow.sklearn.log_model(
        sk_model=rdf,
        artifact_path=artifact_path,
        registered_model_name=MODEL_NAME
    )

    # register model
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=MODEL_VERSION,
        stage=MODEL_STAGE
    )                        