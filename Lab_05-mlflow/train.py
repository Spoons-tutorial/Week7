import mlflow
import collections
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_data(train_dir):
    train = pd.read_csv(train_dir, index_col=["PassengerId"])

    return train

def _encoding(df) -> pd.DataFrame:
    df.loc[df["Sex"] == "male", "Sex"] = 0
    df.loc[df["Sex"] == "female", "Sex"] = 1
    
    return df

def preprocessing(train, features, target):
    train = _encoding(train).fillna(0)
    
    train_x, train_y = train[features], train[target]
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1)

    return train_x, train_y, valid_x, valid_y


def create_model(train_x, train_y):
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    return model


def evaluation(model, valid_x, valid_y):
    pred_y = model.predict(valid_x)
    score = accuracy_score(valid_y, pred_y)
    return score


if __name__ == '__main__':
    # MLflow
    mlflow.set_tracking_uri('http://localhost:5001')
    mlflow.set_experiment('titanic')
    MODEL_PATH = 'titanic_model'
    METRIC = 'accuracy'

    # Directory
    train_dir = "train.csv"

    # features, target
    features = ["Pclass", "Sex", "Fare"]
    target = 'Survived'

    # WorkFlow
    train = load_data(train_dir)
    train_x, train_y, valid_x, valid_y = preprocessing(train, features, target)
    model = create_model(train_x, train_y)
    score = evaluation(model, valid_x, valid_y)

    # logging
    mlflow.log_param("train_dir", train_dir)
    mlflow.log_param("train dataset count", len(train_x))
    mlflow.log_param("target class count", len(set(train_y)))
    mlflow.log_artifact("train.csv")

    mlflow.log_metric(METRIC, score)
    mlflow.sklearn.log_model(model, MODEL_PATH)