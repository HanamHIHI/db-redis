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

df = pd.read_csv("df_final_v6.csv")
name_list = df["name"].unique().tolist()
df_mean_vectors = pd.read_csv("hanam_mean_vectors_100000_v2.csv",header=None)
df_category = pd.read_csv("category_embedding_v6.csv")

category_list = ['해물 요리', '한식당', '일식당', '양식', '고기 요리', '카페', '식당', '디저트',
       '햄버거', '식당 아님', '분식', '치킨', '호프', '피자', '중국집', '베이커리', '아시안 음식',
       '야채 요리', '주류']

print(len(df), len(df_mean_vectors))

schema = [
    NumericField("$.idx", as_name="idx"),
    TextField("$.name", as_name="name"),
    TextField("$.addr", as_name="addr"),
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
]
definition = IndexDefinition(prefix=["restaurant:"], index_type=IndexType.JSON)
res = client.ft("idx:restaurant_vss").create_index(fields=schema, definition=definition)
print(res)

# client.ft("idx:restaurant_vss").create_index(schema)

dict_list = []

print(type(df_mean_vectors.iloc[0]), df_mean_vectors.iloc[0].shape, type(df_mean_vectors.iloc[0][0]), df_mean_vectors.iloc[0][0].shape)
print(type(df_category.iloc[0]), df_category.iloc[0].shape, type(df_category.iloc[0][0]), df_category.iloc[0][0].shape)

for i in range(len(df)):
    row = df.iloc[i]
    # vector = df_mean_vectors.iloc[name_list.index(row["name"])]
    vector = df_mean_vectors.iloc[i]
    if(i % 50 == 0):
        print(i, row["name"])

    if(pd.isna(row["name"]) != True):
        if(pd.isna(row["category3"]) == True or row["category3"]==''):
            # temp_dict = {
            #     "idx": int(row["index"]),
            #     "name": row["name"],
            #     "addr": str(row["position"]).strip('"'),
            #     "dist": int(row["total_distance"]),
            #     "reqtime": int(row["total_time"]),
            #     "category0": '',
            #     "vector": vector.astype(np.float32).tolist(),
            #     "category_vector": df_category.iloc[category_list.index("식당 아님")].astype(np.float32).to_list(),
            # }
            continue
        else:
            temp_dict = {
                "idx": int(row["index"]),
                "name": row["name"],
                "addr": str(row["position"]).strip('"').replace('~', ''),
                "dist": int(row["total_distance"]),
                "reqtime": int(row["total_time"]),
                "category0": row["category3"],
                "vector": vector.astype(np.float32).tolist(),
                "category_vector": df_category.iloc[category_list.index(row["category3"])].astype(np.float32).to_list(),
            }
    else:
        continue

    # json_dict = json.dumps(temp_dict, ensure_ascii=False).encode('utf-8')
    # client.set(str(row["0"], json_dict))
    # print(row["0"], type(row["index"]), type(str(row["index"].astype(np.int_))), len(vector.astype(np.float32).tolist()))
    client.json().set("restaurant:"+str(int(row["index"])), '$', temp_dict)