from sentence_transformers import SentenceTransformer
import torch
import redis
from redis.commands.search.query import Query
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

embeddings = model.encode(["팀점하기 좋은 식당"]).astype(np.float32).tolist()
client = redis.Redis(host="localhost", port=6379, decode_responses=True)

query = (
    Query('*=>[KNN 3 @vector $query_vector AS vector_score]')
     .sort_by('vector_score')
     .return_fields('vector_score', 'idx')
     .dialect(2)
)
res0 = client.ft('idx:restaurant_vss').search(
    query,
    query_params={
      'query_vector': np.array(embeddings, dtype=np.float32).tobytes()
    }
).docs

print(res0)