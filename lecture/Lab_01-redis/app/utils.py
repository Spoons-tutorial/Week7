import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

REDIS_EXPIRE_TIME = 1

import redisai as rai

client = rai.Client(host="localhost", port=6379)


def check_key(model_name):
    return client.exists(model_name)


def load_model(model_path):
    model = joblib.load(model_path)

    return model

def get_model_redis(model_name):
    if client.exists(model_name):
        client.expire(model_name, REDIS_EXPIRE_TIME)
        return pickle.loads(client.get(model_name))
    else:
        return None

def set_model_redis(model, model_name):
    client.set(model_name, pickle.dumps(model))
    client.expire(model_name, REDIS_EXPIRE_TIME)


def set_model_rai(model_path: str, model_name):
    is_set = False
    try:
        model = load_model(model_path=model_path)

        initial_type = [("input", FloatTensorType([None, 4]))]
        onx_model = convert_sklearn(model, initial_types=initial_type)

        client.modelstore(model_name, "onnx", "cpu", onx_model.SerializeToString())
        client.expire(model_name, REDIS_EXPIRE_TIME)

        is_set = True
    except Exception as e:
        is_set = False
        print(f"set_model error: {str(e)}")

    return is_set


def predict_redisai(model_name, input_tensor):
    client.tensorset("input_tensor", input_tensor)
    client.modelrun(
        model_name, ["input_tensor"], ["output_tensor_class", "output_tensor_prob"]
    )
    client.expire(model_name, REDIS_EXPIRE_TIME)

    return client.tensorget("output_tensor_class").item()
