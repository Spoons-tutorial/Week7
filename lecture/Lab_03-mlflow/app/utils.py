import mlflow

mlflow.set_tracking_uri("http://localhost:5000")


def load_rf_clf(model_name: str):
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

    return model
