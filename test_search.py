import json
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import redis
from redis.commands.search.query import Query

import time
import numpy as np
import pandas as pd

print("start setting")
start = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

embeddings = model.encode(["팀점하기 좋은 식당"]).astype(np.float32).tolist()

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
end = time.time()
print(f"{end - start:.5f} sec")

query0 = (
    Query("*=>[KNN 1033 @vector $query_vector AS vector_score]")
     .sort_by('vector_score', asc=False)
     .return_fields('idx', 'vector_score')
     .dialect(2)
)
print("start0")
start = time.time()
res0 = client.ft('idx:restaurant_vss').search(
    query0,
    {
      'query_vector': np.array(embeddings, dtype=np.float32).tobytes()
    }
)
end = time.time()
print(f"{end - start:.5f} sec")

query1 = (
    Query("*=>[KNN 1033 @category_vector $query_category_vector AS category_vector_score]")
     .sort_by('vector_score', asc=False)
     .return_fields('idx', "category_vector_score")
     .dialect(2)
)
print("start1")
start = time.time()
res1 = client.ft('idx:restaurant_vss').search(
    query1,
    {
      'query_vector': np.array(embeddings, dtype=np.float32).tobytes()
    }
)
end = time.time()
print(f"{end - start:.5f} sec")

print("start_sum")
start = time.time()
from collections import Counter
result = dict(Counter(res0)+Counter(res1))

df = pd.DataFrame.from_dict(data=result, orient='index').reset_index()
df['2'] = df['0']+df['1']
top5 = df.sort_values(by='2',ascending=False)[:5]

df_data = pd.read_csv("df_final_v4.csv")
df_final = pd.merge(df_data, top5, how="inner", on="index")
end = time.time()
print(f"{end - start:.5f} sec")