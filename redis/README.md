# Redis

- Redis 는 인메모리 데이터베이스 입니다.
- 사용되는 커맨드는 [공식문서](https://redis.io/commands)에서 자세히 찾아보실 수 있습니다.
- 실습에서는 model을 caching하기위해 redisai를 사용하였습니다.
  ```python
  is_model_exist = redisai_client.exists(REDIS_KEY)

  if not is_model_exist:
      loaded_model = load_rf_clf(model_path)

      initial_type = [("input_tensor", FloatTensorType([None, 4]))]

      onx_model = convert_sklearn(loaded_model, initial_types=initial_type)

      redisai_client.modelstore(
          key=REDIS_KEY, 
          backend='onnx', 
          device='cpu', 
          data=onx_model.SerializeToString()
      )
  ```
  - 실습에서는 redisai를 사용하여 모델캐싱을 진행하였지만 모델캐싱 뿐만아니라 자주 요청되는 데이터에대해 redis에 응답을 캐싱해두고 모델 prediction을 거치지않고 캐싱해둔 결과를 응답으로 반환할 수 있습니다.
  - 모델을 캐싱할 때 사용된 것은 아래와 같습니다.
    - `redisai command with python`: [modelstore](https://redisai-py.readthedocs.io/en/latest/api.html?highlight=modelstore#redisai.Client.modelstore)가 사용되었습니다.
    - `REDIS_KEY`: 해당 Key로 model을 저장하거나 조회할 때 사용됩니다.
    - `convert_sklearn`: 공식문서에 따르면 `modelstore`에서 허용되는 backend값은 TF, TORCH, TFLITE, ONNX 입니다. sklearn은 지원되지 않기때문에 onnx형태로 바꾸기 위해 사용되었습니다.

- onnx로 모델을 저장하였기 때문에 inference에 기존처럼 predict를 사용할 수 없습니다.
  ```python
  redisai_client.modelrun(key=REDIS_KEY,
                      inputs=['input_tensor'],
                      outputs=['output_tensor_class', 'output_tensor_prob'])

  result = int(redisai_client.tensorget('output_tensor_class'))
  ```
  - `key`: Model의 key값입니다.
  - `inputs`: redisai_client.tensorset()메서드를 이용하여 미리 저장해둔 데이터의 이름입니다.
  - `outputs`: 출력값을 저장할 키입니다. 키값이 이미 존재하면 새로운 값으로 덮어씁니다.

  