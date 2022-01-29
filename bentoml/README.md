# BentoML

## Lab1

- [BentoML](https://www.bentoml.ai/)은 'Model Serving Made Easy' 라는 슬로건을 보면 알 수 있듯이 모델 서빙에 강점이 있는 라이브러리 입니다.
- [BentoService](https://docs.bentoml.org/en/0.13-lts/concepts.html#creating-bentoservice)를 상속받은 클래스를 작성함으로써 prediction service를 만듭니다.
  ```python
    @artifacts([SklearnModelArtifact("mtcars_rf")])
    class MtcarsAPI(BentoService):
        def pred_mtcars(self, df: pd.DataFrame):
            return self.artifacts.mtcars_rf.predict(df)
  ```
  - 여기서 `mtcars_rf` 이름은 모델 내부에서 사용할 이름입니다.
  - `SklearnModelArtifact('이름')` 에서의 이름과 `self.artifacts.이름.predict(df)`의 이름은 같아야합니다.

- [pack](https://docs.bentoml.org/en/0.13-lts/concepts.html#packaging-model-artifacts)을 통해 BentoService에서 모델을 사용합니다.
  - `pack('이름', 모델)` 에서 이름은 MtcarsAPI를 작성할 때 적은 이름과 같게합니다. (주입)

## Lab2

- 앞서 실습한 MLFlow와 BentoML을 함께 사용하는 단계입니다.
- Lab1에서는 BentoService에 모델을 pack할 때 코드상에서 모델을 만들어 바로 주입하였습니다.
- Lab2에서는 BentoService에 모델을 pack할 때 mlflow에서 모델을 불러와 주입합니다.