import argparse
import mlflow
import pandas as pd

if __name__ == '__main__':
    # MLflow
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('titanic')
    MODEL_PATH = 'titanic_model'
    METRIC = 'accuracy'

    # 호출당시 인자값을 설정할 수 있습니다.
    parser = argparse.ArgumentParser(description="Predict titanic surviver")

    parser.add_argument('--pclass', type=int, default=2, choices=range(1,4))
    parser.add_argument('--sex', type=int, default=0, choices=range(0,2))
    parser.add_argument('--fare', type=float, default=300.0)

    args = parser.parse_args()

    # experiment이름으로 모든 runs를 찾습니다.
    exp = mlflow.get_experiment_by_name('titanic')
    exp_id = exp.experiment_id
    runs = mlflow.search_runs([exp_id])

    # metric에 대해 가장 나은 모델을 가져옵니다.
    best_score = runs[f'metrics.{METRIC}'].max()
    best_run = runs[runs[f'metrics.{METRIC}'] == best_score]
    best_run_id = best_run.run_id.values[0]

    titanic_target_names = {
        0: 'Dead',
        1: 'Survived'
    }
    test_x = pd.DataFrame({
        "Pclass": args.pclass, "Sex": args.sex, "Fare": args.fare
    }, index=[0])

    # 모델 uri를 바탕으로 모델을 load 해옵니다.
    best_model_uri = f'runs:/{best_run_id}/{MODEL_PATH}'
    loaded_model = mlflow.sklearn.load_model(best_model_uri)

    result = loaded_model.predict(test_x)
    print(titanic_target_names[result[0]])