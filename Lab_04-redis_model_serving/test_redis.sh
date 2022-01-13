docker exec -it redis-dev\
 redis-cli
# docker로 띄운 redis-cli에 접속합니다.
# (nil)이 출력될 경우 캐싱되지 않은 상태입니다.
# get redis_caching_model 를 입력하여 caching된 결과를 확인합니다.