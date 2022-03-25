from bentoml_service import IrisAPI, IrisAPIMlflow
from utils import load_rf_clf

# 일반적인 BentoML Service 생성 프로세스
model = load_rf_clf('assets/random_forest.pkl')

# BentoML Service 객체 생성
bento_service = IrisAPI()

# Service에 모델 Pack
bento_service.pack('iris_rf', model)

# Service 저장
bento_service.save()


# Mlflow에서 모델을 불러오도록 Cutomize된 BentoML Service 생성 프로세스
## bento_service = IrisAPIMlflow()

# Service 시작 단계에서 모델을 불러오기 때문에 모델을 pack할 필요가 없다
## bento_service.save()