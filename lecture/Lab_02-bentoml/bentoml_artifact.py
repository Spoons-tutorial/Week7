import os
import mlflow
from bentoml.service.env import BentoServiceEnv
from bentoml.service.artifacts import BentoServiceArtifact
from bentoml.exceptions import MissingDependencyException

DEFAULT_PICKLE_EXTENSION = ".pkl"

mlflow.set_tracking_uri("http://localhost:5001")


def _import_joblib_module():
    try:
        import joblib
    except ImportError:
        joblib = None

    if joblib is None:
        try:
            from sklearn.externals import joblib
        except ImportError:
            pass

    if joblib is None:
        raise MissingDependencyException(
            "sklearn module is required to use SklearnModelArtifact"
        )

    return joblib


# https://github.com/bentoml/BentoML/blob/0.13-LTS/bentoml/frameworks/sklearn.py
class MlflowArtifact(BentoServiceArtifact):
    def __init__(self, name, model_name):
        super(MlflowArtifact, self).__init__(name)
        self._model_name = model_name
        self._model = None
        self._packed = True

    def _model_file_path(self, base_path):
        return os.path.join(base_path, self.name + DEFAULT_PICKLE_EXTENSION)

    def pack(self):
        pass

    # BentoML Service 실행 시 Mlflow 에서 모델을 불러오는 부분
    def load(self, path):
        joblib = _import_joblib_module()
        model_file_path = self._model_file_path(path)

        if not os.path.exists(model_file_path):
            self._model = mlflow.sklearn.load_model(
                f"models:/{self._model_name}/Production"
            )
            self.save(path)
            print(model_file_path)
        else:
            self._model = joblib.load(model_file_path)

    def get(self):
        return self._model

    # Mlflow에서 정상적으로 모델을 불러왔다면, 지속적으로 사용할 수 있도록 로컬에 해당 모델을 저장하는 부분
    def save(self, dst):
        if self._model is not None:
            joblib = _import_joblib_module()
            joblib.dump(self._model, self._model_file_path(dst))

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(["scikit-learn"])
