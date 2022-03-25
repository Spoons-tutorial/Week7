import mlflow
import joblib

# mlflow에 모델을 저장하는 예제

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLFLOW_TEST")

model = joblib.load("assets/random_forest.pkl")

mlflow.start_run()

mlflow.sklearn.log_model(model, "model", registered_model_name="iris_rf_model")

mlflow.end_run()
