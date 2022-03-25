import pandas as pd
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput

from bentoml_artifact import MlflowArtifact
from bentoml.frameworks.sklearn import SklearnModelArtifact


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact("iris_rf")])  # Scikit-learn 기본 Artifact 사용
class IrisAPI(BentoService):
    @api(input=DataframeInput(), route="v1/iris/predict", batch=True)
    def predict(self, df: pd.DataFrame):
        """
        Iris inference API
        """

        return self.artifacts.iris_rf.predict(df)


@env(infer_pip_packages=True)
@artifacts(
    [MlflowArtifact("iris_rf", "iris_rf_model")]
)  # Mlflow에서 모델을 불러오는 Custom Artifact 사용
class IrisAPIMlflow(BentoService):
    @api(input=DataframeInput(), route="v1/iris-mlflow/predict", batch=True)
    def predict(self, df: pd.DataFrame):
        """
        Iris inference API
        """

        return self.artifacts.iris_rf.predict(df)
