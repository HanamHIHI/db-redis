import redis

# Redis 클라이언트 연결
r = redis.Redis(host='localhost', port=6379)

# 인덱스에 포함된 모든 문서 ID 가져오기
results = r.execute_command('flushall')

# # 모든 문서 삭제
# for doc_id in results[1:]:  # 첫 번째 항목은 결과 개수, 이후부터가 문서 ID
#     r.delete(doc_id)
