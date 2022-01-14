docker exec -it redis-dev\
 redis-cli
# docker로 띄운 redis-cli에 접속합니다.
# (nil)이 출력될 경우 캐싱되지 않은 상태입니다.
# get redis_caching_model 를 입력하여 caching된 결과를 확인합니다.

# set Hello World!
# get Hello
    # World!
# del Hello
# exists Hello
    # 0

# hset users email1 choonsik@gmail.com
# hset users email2 lion@gmail.com
# keys *
    # users
# hget users email1
    # "choonsik@gmail.com"
# hgetall users
    # 1) "email1"
    # 2) "choonsik@gmail.com"
    # 3) "email2"
    # 4) "lion@gmail.com"

# set Hello World!
# get Hello
    # "World!"
# expire Hello 10
    # 10초후 (nil)
# ttl Hello
    # 남은시간~

# 왼쪽에서 차례로 넣음 순서는 
# lpush name_list choonsik lion
# lpush name_list mooji apeach
# rpush name_list 춘식 라이언
# rpush name_list 무지 어피치
# lrange name_list 0 -1
    # apeach
    # mooji
    # lion
    # choonsik
    # 춘식
    # 라이언
    # 무지
    # 어피치

# 한글이 깨지는거는 접속 커맨드를
# docker exec -it redis-dev redis-cli --raw 이렇게 수정하면 됨