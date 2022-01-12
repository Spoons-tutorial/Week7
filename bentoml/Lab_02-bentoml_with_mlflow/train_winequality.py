import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import mlflow
from mlflow.tracking import MlflowClient

basedir = os.getenv("HOME")

exp_name = "wine_quality"
artifact_path = "winequal_model"

mlflow.set_tracking_uri("http://localhost:8000")
mlflow.set_experiment(exp_name)

client = MlflowClient()
exp_id = client.get_experiment_by_name(exp_name).experiment_id


with mlflow.start_run(experiment_id = exp_id):    
    data = pd.read_csv(f"{basedir}/Pipeline/bentoml/data/winequality-white.csv")

    y = "quality"
    X_data = data.loc[:, data.keys() != y]
    y_data = data[y]

    std = preprocessing.StandardScaler()
    X_data = std.fit_transform(X_data)

    le = preprocessing.LabelEncoder()
    y_data = le.fit_transform(y_data)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

    train_data = xgb.DMatrix(data = X_train, label = y_train)

    params = {"gamma": 0.1, 
              "max_depth": 3, 
              "objective":"multi:softmax",
              "num_class": 10,
              "eval_metric": "merror",
              "seed": 2022}
    xgb_clf = xgb.train(params = params,
                        dtrain = train_data,
                        num_boost_round = 50)

    result = xgb_clf.predict(xgb.DMatrix(X_test))
    accuracy = sum(result == y_test)/len(result)

    mlflow.log_params({"gamma":params["gamma"],
                       "max_depth":params["max_depth"]})
    mlflow.log_metric(key = "accuracy", value = accuracy)
    mlflow.xgboost.log_model(xgb_model=xgb_clf,
                             artifact_path=artifact_path)