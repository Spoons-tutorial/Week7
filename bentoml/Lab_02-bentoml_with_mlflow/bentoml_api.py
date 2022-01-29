from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.frameworks.xgboost import XgboostModelArtifact

import pandas as pd
from sklearn import preprocessing
import xgboost as xgb


@env(pip_packages = ["bentoml==0.13.1",
                     "pandas==1.3.5",
                     "scikit-learn==1.0.1",
                     "xgboost==1.4.2"])
@artifacts([SklearnModelArtifact("mtcars_rf"),
            XgboostModelArtifact("winequal_xgb")])
class ModelApi(BentoService):
    mtcars_apidoc = """
    mtcars inference API


    Inputs:
        cyl:	Number of cylinders
        disp:	Displacement (cu.in.)
        hp:	    Gross horsepower
        drat:	Rear axle ratio
        wt:	    Weight (1000 lbs)
        qsec:	1/4 mile time
        vs:	    Engine (0 = V-shaped, 1 = straight)
        am:	    Transmission (0 = automatic, 1 = manual)
        gear:	Number of forward gears
        carb:	Number of carburetors
    
    Returns:
        mpg:    Miles/(US) gallon
    """
    mtcars_input_exam = [{'cyl': 6,
                        'disp': 160.0,
                        'hp': 110,
                        'drat': 3.90,
                        'wt': 2.620,
                        'qsec': 16.46,
                        'vs': 0,
                        'am': 1,
                        'gear': 4,
                        'carb': 4}]
    

    @api(input = DataframeInput(
            http_input_example = mtcars_input_exam), 
         api_doc = mtcars_apidoc,
         route = 'v1/mtcars',
         batch = True)
    def pred_mtcars(self, df: pd.DataFrame):
        return self.artifacts.mtcars_rf.predict(df)


    @api(input = DataframeInput(
            http_input_example = [{'fixed acidity': 5.8,
                                'volatile acidity': 0.32,
                                'citric acid': 0.2,
                                'residual sugar': 2.6,
                                'chlorides': 0.027,
                                'free sulfur dioxide': 17.0,
                                'total sulfur dioxide': 123.0,
                                'density': 0.98936,
                                'pH': 3.36,
                                'sulphates': 0.78,
                                'alcohol': 13.9}]), 
         route = 'v1/winequal',
         batch = True)
    def pred_winequal(self, df: pd.DataFrame):
        """
        winequality inference API


        Inputs:
            01 - fixed acidity
            02 - volatile acidity
            03 - citric acid
            04 - residual sugar
            05 - chlorides
            06 - free sulfur dioxide
            07 - total sulfur dioxide
            08 - density
            09 - pH
            10 - sulphates
            11 - alcohol
        
        Returns:
            12 - quality (score between 0 and 10)
        """

        std = preprocessing.StandardScaler()
        df = std.fit_transform(df)
        df = xgb.DMatrix(df)
        return self.artifacts.winequal_xgb.predict(df)