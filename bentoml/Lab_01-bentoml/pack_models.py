import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from bentoml_api import MtcarsAPI

basedir = os.getenv("HOME")

data = pd.read_csv(f"{basedir}/Week7/bentoml/data/mtcars.csv")
data = data.drop(labels = 'Unnamed: 0', axis = 1) # 필요없는 컬럼 제거
X_data = data.loc[:, data.keys() != 'mpg'] # target, label 분리
y_data = data['mpg']


rdf = RandomForestRegressor(random_state=2022)
rdf.fit(X_data.values, y_data.values)


bento_service = MtcarsAPI()
bento_service.pack("mtcars_rf", rdf)

saved_path = bento_service.save()