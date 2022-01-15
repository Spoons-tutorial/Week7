import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.tracking import MlflowClient

basedir = os.getenv("HOME")


if __name__ == '__main__':
    data = pd.read_csv("./mtcars.csv")
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
    joblib.dump(rdf, './rf_reg')
    a = joblib.load('./rf_reg')
    print(a)
    