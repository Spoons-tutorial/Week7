import json
import random
from locust import HttpUser, TaskSet, task, between

class User(HttpUser):
    
    @task
    def predict_iris(self):
        self.client.post("/iris", json={
            "sepal_length": 4.8,
            "sepal_width": 4.1,
            "petal_length": 3.3,
            "petal_width": 1.7,
        }, params={
            'model_name': "rf_clf_model_0106"
        })

    def on_start(self):
        print("START LOCUST")

    def on_stop(self):
        print("STOP LOCUST")