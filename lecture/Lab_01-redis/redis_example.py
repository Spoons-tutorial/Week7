import redis

# Redis Client 생성
redis_client = redis.StrictRedis(host='localhost', port=6379)

# Exist Key: 'key1' then return False
redis_client.exitsts('key1')

# Set Key: 'key1' - Value: 1
redis_client.set('key1', 1)

# Exist Key: 'key1' then return True
redis_client.exists('key1')

# Get Key: 'key1' then return 1
redis_client.get('key1')

# Set expire time for 1 second
redis_client.expire('key', 1)


import redisai as rai

# # Redis-AI Client 생성
redisai_client = rai.Client(host="localhost", port=6379)

# # Exist Key: 'input_tensor' then return False
redisai_client.exists('input_tensor')

# Set Tensor Key: 'input_tensor' - Value: Numpy Array | List | Tuple
redisai_client.tensorset('input_tensor', [[1,2,3,4]])

# Exist Key: 'input_tensor' then return True
redisai_client.exists('input_tensor')

# Get Tensor Key: 'input_tensor' then return [[1,2,3,4]]
redisai_client.tensorget('input_tensor')

# Set expire time for 1 second
redisai_client.expire('input_tensor', 1)

# Store Model Key: 'random_forest' - Value: ONNX Model(RandomForestClassifier)
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
initial_type = [("input", FloatTensorType([None, 4]))]
onx_model = convert_sklearn(model, initial_types=initial_type)

redisai_client.modelstore(key='random_forest', 
                          backend='onnx', 
                          device='cpu', 
                          data=onx_model.SerializeToString())

# Run Model Model Key: 'random_forest', Input Key: ['input_tensor']
# Output Key: ['output_tensor_class', 'output_tensor_prob'])
redisai_client.modelrun(key='random_forest',
                        inputs=['input_tensor'],
                        outputs=['output_tensor_class', 'output_tensor_prob'])