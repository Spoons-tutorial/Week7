import argparse
import mlflow
import pandas as pd

if __name__ == '__main__':
    # MLflow
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('titanic')
    METRIC = 'accuracy'
    
    MODEL_NAME = 'sk-learn-rf-clf-model'
    MODEL_VERSION = 1 # Way1
    MODEL_STAGE = 'Production' # Way2

    # 호출당시 인자값을 설정할 수 있습니다.
    parser = argparse.ArgumentParser(description="Predict titanic surviver")

    parser.add_argument('--pclass', type=int, default=2, choices=range(1,4))
    parser.add_argument('--sex', type=int, default=0, choices=range(0,2))
    parser.add_argument('--fare', type=float, default=300.0)

    args = parser.parse_args()

    titanic_target_names = {
        0: 'Dead',
        1: 'Survived'
    }
    test_x = pd.DataFrame({
        "Pclass": args.pclass, "Sex": args.sex, "Fare": args.fare
    }, index=[0])

    # Way1: 모델 name과 version을 기반으로 uri를 구성합니다.
    model_uri1 = f'models:/{MODEL_NAME}/{MODEL_VERSION}'
    loaded_model_by_version = mlflow.sklearn.load_model(model_uri1)

    # Way2: 모델 name과 stage를 기반으로 uri를 구성합니다.
    model_uri2 = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    loaded_model_by_stage = mlflow.sklearn.load_model(model_uri2)

    result1 = loaded_model_by_version.predict(test_x)
    result2 = loaded_model_by_stage.predict(test_x)
    
    print(titanic_target_names[result1[0]])
    print(titanic_target_names[result2[0]])