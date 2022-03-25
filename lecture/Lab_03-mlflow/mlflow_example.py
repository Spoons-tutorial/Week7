import joblib

import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('MLFLOW_TEST_EXPERIMENT')
# mlflow.create_experiment('MLFLOW_NEW_TEST_EXPERIMENT')


if __name__ == '__main__':
    model = joblib.load('assets/random_forest.pkl')

    params = {"n_estimators": 50,
              "max_depth": 5,}

    metrics = {"accuracy": 0.765,
               "logloss": 0.654,
               "auc": 0.7}
    
    with mlflow.start_run(tags={'version': '1.0.0'}):
        mlflow.log_params(params)
        mlflow.log_param("criterion", 'gini')

        mlflow.log_metrics(metrics)
        mlflow.log_metric('recall', 0.7)

        mlflow.sklearn.log_model(model, 
                                    'rf_model',
                                    registered_model_name='random_forest')

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('MLFLOW_TEST_EXPERIMENT')

    client = MlflowClient()

    run = client.get_run('acf319e9531043908fb6bc305cb615a9')

    metrics = run.data.metrics
    params  = run.data.params
    tags    = run.data.tags

    lifecycle_stage = run.info.lifecycle_stage