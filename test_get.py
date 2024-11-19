import redis
import json

# Redis 클라이언트 연결
r = redis.Redis(host='localhost', port=6379)

# # 벡터가 제대로 저장되었는지 확인
# print(len(r.json().get("restaurant:1", '$.vector')), len(r.json().get("restaurant:1", '$.cv')))

# info = r.ft("idx:restaurant_vss").info()
# num_docs = info["num_docs"]
# indexing_failures = info["hash_indexing_failures"]

# print(num_docs, indexing_failures)

print(r.json().get("restaurant:830", "$.category_vector"))