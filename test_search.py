from sentence_transformers import SentenceTransformer
from sentence_transformers import util
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

try:
    model_state_dict = torch.load("basic_model_1000" +  ".pt", map_location=device)
    try:
        model.load_state_dict(model_state_dict)

    except RuntimeError:
        print("E1")
        try:
            model.load_state_dict(model_state_dict, strict=False)
        except:
            pass

except FileNotFoundError:
    print("E0")
    try:
        model_state_dict = torch.load("basic_model_1000" +  ".pt", map_location=device)
        try:
            model.load_state_dict(model_state_dict)

        except RuntimeError:
            print("E1")
            try:
                model.load_state_dict(model_state_dict, strict=False)
            except:
                pass
    except:
        pass

print("model loading complete.")

embeddings = model.encode(["중국집"]).astype(np.float32).tolist()
chinese = model.encode(["중국집"])
beer = model.encode(["호프"])
boonsik = model.encode(["분식"])
seafood = model.encode(["해물 요리"])

print("중국집-chinese", util.cos_sim(np.array(embeddings[0], dtype=np.float32), chinese))
print("중국집-beer", util.cos_sim(np.array(embeddings[0], dtype=np.float32), beer))
print("중국집-boonsik", util.cos_sim(np.array(embeddings[0], dtype=np.float32), boonsik))
print("중국집-seafood", util.cos_sim(np.array(embeddings[0], dtype=np.float32), seafood))

print(type(embeddings), len(embeddings), embeddings[0][0])

client = redis.Redis(host="localhost", port=6379, decode_responses=True)
end = time.time()
print(f"{end - start:.5f} sec")

print("start0")
start = time.time()
query0 = (
    Query("((-@category0:\"식당 아님\") (@reqtime:[0 600]))=>[KNN 15 @vector $query_vector AS vector_score]")
     .sort_by('vector_score')
     .return_fields('idx', 'vector_score')
     .dialect(2)
     .paging(0, 1034)
)
res0 = client.ft('idx:restaurant_vss').search(
    query0,
    {
      'query_vector': np.array(embeddings, dtype=np.float32).tobytes()
    }
).docs
# print(res0)
str_res0 = str(res0).replace("Result{15 total, docs: ", '').replace("Document", '')
dict_res0 = eval(str_res0)
end = time.time()
print(f"{end - start:.5f} sec")

print("start1")
start = time.time()
query1 = (
    Query("((-@category0:\"식당 아님\") (@reqtime:[0 600]))=>[KNN 15 @category_vector $query_vector AS category_vector_score]")
     .sort_by('category_vector_score')
     .return_fields('idx', "category_vector_score")
     .dialect(2)
     .paging(0, 1034)
)
res1 = client.ft('idx:restaurant_vss').search(
    query1,
    {
      'query_vector': np.array(embeddings, dtype=np.float32).tobytes()
    }
).docs
# print(res1)
str_res1 = str(res1).replace("Result{15 total, docs: ", '').replace("Document", '')
dict_res1 = eval(str_res1)
end = time.time()
print(f"{end - start:.5f} sec")

print("start_sum")
start = time.time()
# from collections import Counter
# result = dict(Counter(dict_res0)+Counter(dict_res1))

df0 = pd.DataFrame.from_dict(data=dict_res0).reset_index()
print(df0.head(), len(df0))

df1 = pd.DataFrame.from_dict(data=dict_res1).reset_index()
print(df1.head(), len(df1))

df2 = pd.merge(left=df0, right=df1, how="inner", on="idx")[["idx", "vector_score", "category_vector_score"]]
df2["mixed_score"] = df2["vector_score"].astype("float")*df2["category_vector_score"].astype("float")
df2 = df2.sort_values(by="mixed_score", ascending=True)
print(df2.head(), len(df2))



# df['2'] = df['0']+df['1']
# top5 = df.sort_values(by='2',ascending=False)[:5]

# df_data = pd.read_csv("df_final_v4.csv")
# df_final = pd.merge(df_data, top5, how="inner", on="index")
# end = time.time()
# print(f"{end - start:.5f} sec")

# print(df_final)