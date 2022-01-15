from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.adapters.json_output import JsonOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact

import pandas as pd


@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact("mtcars_rf")])
class MtcarsAPI(BentoService):
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
         output = JsonOutput(),
         api_doc = mtcars_apidoc,
         batch = True)
    def pred_mtcars(self, df: pd.DataFrame):
        return self.artifacts.mtcars_rf.predict(df)