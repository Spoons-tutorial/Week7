import json
import random
from locust import HttpUser, TaskSet, task, between

class User(HttpUser):
    
    @task
    def predict_iris(self):
        wait_time = between(1, 2)

        self.client.post("/", json={
                'cyl': 6,
                'disp': 160.0,
                'hp': 110,
                'drat': 3.90,
                'wt': 2.620,
                'qsec': 16.46,
                'vs': 0,
                'am': 1,
                'gear': 4,
                'carb': 4
        })

    def on_start(self):
        print("START LOCUST")

    def on_stop(self):
        print("STOP LOCUST")