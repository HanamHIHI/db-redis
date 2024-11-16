"""
Code samples for vector database quickstart pages:
    https://redis.io/docs/latest/develop/get-started/vector-database/
"""
import numpy as np
import pandas as pd
from redis.commands.json.path import Path
import redis
from redis.commands.search.field import (
    NumericField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

client = redis.Redis(host="localhost", port=6379, decode_responses=True)

df = pd.read_csv("df_final_v4.csv")
df_mean_vectors = pd.read_csv("hanam_mean_vectors_100000.csv",header=None)
df_category = pd.read_csv("category_embedding.csv")

category_list = ['해물', '한식당', '돈가스', '양식', '고기', '카페', '식당', '일식당', '간식',
       '햄버거', '까페', '식당 아님', '분식', '치킨', '호프', '피자', '일식집', '중국집', '베이커리',
       '아시안 음식', '야채', '샤브샤브', '주류', '해장국', '디저트', '뷔페']

print(len(df), len(df_mean_vectors))

schema = (
    NumericField("$.idx", as_name="idx"),
    TextField("$.name", no_stem=True, as_name="name"),
    TextField("$.addr", no_stem=True, as_name="addr"),
    NumericField("$.dist", as_name="dist"),
    NumericField("$.reqtime", as_name="reqtime"),
    TextField("$.category0", as_name="category0"),
    VectorField(
        "$.vector",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": 768,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
    VectorField(
        "$.category_vector",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": 768,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="category_vector",
    ),
    
)
definition = IndexDefinition(prefix=["restaurant:"], index_type=IndexType.JSON)
res = client.ft("idx:restaurant_vss").create_index(fields=schema, definition=definition)
print(res)

dict_list = []

for i in range(len(df)):
    row = df.iloc[i]
    vector = df_mean_vectors.iloc[i]

    if(pd.isna(row["name"]) != True):
        if(pd.isna(row["category3"]) == True):
            temp_dict = {
                "idx": int(row["index"]),
                "name": row["name"],
                "addr": str(row["position"]).strip('"'),
                "dist": int(row["total_distance"]),
                "reqtime": int(row["total_time"]),
                "category0": '',
                "vector": vector.astype(np.float32).tolist(),
                "category_vector": df_category.iloc[category_list.index("식당 아님")].astype(np.float32).tolist(),
            }
        else:
            temp_dict = {
                "idx": int(row["index"]),
                "name": row["name"],
                "addr": str(row["position"]).strip('"').replace('~', ''),
                "dist": int(row["total_distance"]),
                "reqtime": int(row["total_time"]),
                "category0": row["category3"],
                "vector": vector.astype(np.float32).tolist(),
                "category_vector": df_category.iloc[category_list.index(row["category3"])].astype(np.float32).tolist(),
            }
    else:
        continue

    # json_dict = json.dumps(temp_dict, ensure_ascii=False).encode('utf-8')
    # client.set(str(row["0"], json_dict))
    # print(row["0"], type(row["index"]), type(str(row["index"].astype(np.int_))), len(vector.astype(np.float32).tolist()))
    client.json().set("restaurant:"+str(int(row["index"])), Path.root_path(), temp_dict)