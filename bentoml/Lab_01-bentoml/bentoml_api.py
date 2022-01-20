from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

import pandas as pd


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact("mtcars_rf")])
class MtcarsAPI(BentoService):
    
    @api(input = DataframeInput(), 
         batch = True)
    def pred_mtcars(self, df: pd.DataFrame):
        """
        mtcars inference API
        """
        return self.artifacts.mtcars_rf.predict(df)