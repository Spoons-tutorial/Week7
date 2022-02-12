# Redis

- Redis 는 인메모리 데이터베이스 입니다.
- 사용되는 커맨드는 [공식문서](https://redis.io/commands)에서 자세히 찾아보실 수 있습니다.
- 실습에서는 model을 caching하기위해 redisai를 사용하였습니다.
  ```python
  REDIS_KEY = 'redis_caching_model'

  try:
    onx_model = con.modelget(REDIS_KEY)
    onx_model = onx_model['blob']
  except redis.exceptions.ResponseError:            
    model = load_rf_clf(model_path)
    initial_type = [('float_input', FloatTensorType([1,4]))]
    onx_model = convert_sklearn(model, initial_types=initial_type).SerializeToString()
    con.modelset(REDIS_KEY, 'onnx', 'cpu', onx_model)
  ```
  - 실습에서는 redisai를 사용하여 모델캐싱을 진행하였지만 모델캐싱 뿐만아니라 자주 요청되는 데이터에대해 redis에 응답을 캐싱해두고 모델 prediction을 거치지않고 캐싱해둔 결과를 응답으로 반환할 수 있습니다.
  - 모델을 캐싱할 때 사용된 것은 아래와 같습니다.
    - `redisai command with python`: [modelget](https://redisai-py.readthedocs.io/en/latest/api.html#redisai.Client.modelget), [modelset](https://redisai-py.readthedocs.io/en/latest/api.html#redisai.Client.modelset) 이 사용되었습니다.
    - `REDIS_KEY`: 해당 Key로 model을 저장하거나 조회할 때 사용됩니다.
    - `convert_sklearn`: 공식문서에 따르면 `modelset`에서 허용되는 backend값은 TF, TORCH, TFLITE, ONNX 입니다. sklearn은 지원되지 않기때문에 onnx형태로 바꾸기 위해 사용되었습니다.

- onnx로 모델을 저장하였기 때문에 inference에 기존처럼 predict를 사용할 수 없습니다.
  ```python
    sess = rt.InferenceSession(onx_model)
    test = np.array([*iris_info.dict().values()]).reshape(1,-1).astype(np.float32)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    result = sess.run([label_name], {input_name: test})[0][0]
  ```
  - onnxruntime `InferenceSession`의 첫번째 인자로 path_or_bytes가 가능합니다.
    - 실습코드에서는 onnx model을 그대로 넣어주었지만 onnx모델 경로를 적어주어도 됩니다.
  - `label_name`과 `input_name`은 string type으로 입력하시면 됩니다.
  - `run`: predict와 동일하게 inference를 수행하는 method입니다.
  