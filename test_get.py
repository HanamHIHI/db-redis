import redis
import json

# Redis 클라이언트 연결
r = redis.Redis(host='localhost', port=6379)

retrieved_data = r.json().get("restaurant:345")
print(retrieved_data)