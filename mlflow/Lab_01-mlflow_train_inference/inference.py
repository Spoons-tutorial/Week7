import argparse
import mlflow
import pandas as pd

if __name__ == '__main__':
    # MLflow
    mlflow.set_tracking_uri('http://localhost:5001')
    mlflow.set_experiment('titanic')
    MODEL_PATH = 'titanic_model'
    METRIC = 'accuracy'

    # 값 받기
    parser = argparse.ArgumentParser(description="Predict titanic surviver")

    parser.add_argument('--Pclass', type=int)
    parser.add_argument('--Sex', type=int)
    parser.add_argument('--Fare', type=float)

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
    test_x = pd.DataFrame({"Pclass": [2], "Sex": [0], "Fare": [300.0]})

    # 모델 uri를 바탕으로 모델을 load 해옵니다.
    best_model_uri = f'runs:/{best_run_id}/{MODEL_PATH}'
    loaded_model = mlflow.sklearn.load_model(best_model_uri)

    result = loaded_model.predict(test_x)
    print(titanic_target_names[result[0]])