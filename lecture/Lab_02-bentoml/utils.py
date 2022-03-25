import joblib


def load_rf_clf(model_path):
    model = joblib.load(model_path)

    return model
