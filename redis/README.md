# Redis

- Redis 는 인메모리 데이터베이스 입니다.
- 사용되는 커맨드는 [공식문서](https://redis.io/commands)에서 자세히 찾아보실 수 있습니다.
- 실습에서는 model을 caching하기위해 사용되었습니다.
  ```python
        ### redis
        REDIS_KEY = 'redis_caching_model'
        RESET_SEC = 100

        model = client.get(REDIS_KEY)

        if model:
            # reids에 해당 키가 존재하는 경우입니다. == 모델이 redis에 존재
            # 모델을 load해와 deserialize하고 만료시간을 갱신합니다.
            model = pickle.loads(model)
            client.expire(REDIS_KEY, RESET_SEC)
        else:
            # redis에 해당키가 존재하지 않는 경우입니다. == 모델이 redis에 없음
            # 모델을 기존 방식대로 load해온 뒤에 redis엥 해당모델을 저장합니다.
            model = load_rf_clf(model_path)
            client.set(
                REDIS_KEY, pickle.dumps(model), datetime.timedelta(seconds=RESET_SEC)
            )
        ###
  ```
  - 모델캐싱 뿐만아니라 자주 요청되는 데이터에대해 응답을 캐싱해두고 모델 prediction을 거치지않고 캐싱해둔 결과를 응답으로 반환할 수 있습니다.
  - 모델을 캐싱할 때 사용된 것은 아래와 같습니다.
    - `redis command with python`: [get](https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.cluster.RedisClusterCommands.get), [set](https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.core.CoreCommands.set), [expire](https://redis-py.readthedocs.io/en/stable/commands.html#redis.commands.cluster.RedisClusterCommands.expire)이 사용되었습니다.
    - `REDIS_KEY`: 해당 Key로 value를 저장하거나 조회할 때 사용됩니다.
    - `pickle.dumps(model)`: [redis에서 지원되는 data type](https://redis.io/topics/data-types-intro)으로 변경하기위해 dumps를 통해 model을 바꾸어줍니다.
    - `RESET_SEC`: expire 시간을 설정하기 위해 사용되었습니다. 시간이 지나면 키가 자동적으로 삭제됩니다.