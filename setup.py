"""
Code samples for vector database quickstart pages:
    https://redis.io/docs/latest/develop/get-started/vector-database/
"""

import json
import time

import numpy as np
import pandas as pd
import requests
from redis.commands.json.path import Path
import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
client = redis.Redis(host="localhost", port=6379, decode_responses=True)

df = pd.read_csv("df_final_v4.csv")
df_mean_vectors = pd.read_csv("hanam_mean_vectors_100000.csv",header=None)

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
)
definition = IndexDefinition(prefix=["restaurant:"], index_type=IndexType.JSON)
res = client.ft("idx:restaurant_vss").create_index(fields=schema, definition=definition)
print(res)

dict_list = []

for i in range(len(df)):
    row = df.iloc[i]
    vector = df_mean_vectors.iloc[i]

    if(pd.isna(row["name"]) != True):
        temp_dict = {
            "idx": int(row["index"]),
            "name": row["name"],
            "addr": str(row["position"]).strip('"').replace('~', ''),
            "dist": int(row["total_distance"]),
            "reqtime": int(row["total_time"]),
            "category0": row["category3"],
            "vector": vector.astype(np.float32).tolist(),
        }
        if(pd.isna(row["category3"]) == True):
            temp_dict = {
            "idx": int(row["index"]),
            "name": row["name"],
            "addr": str(row["position"]).strip('"'),
            "dist": int(row["total_distance"]),
            "reqtime": int(row["total_time"]),
            "category0": '',
            "vector": vector.astype(np.float32).tolist(),
        }
    else:
        temp_dict = {
            "idx": int(row["index"]),
            "name": '',
            "addr": '',
            "dist": -1,
            "reqtime": -1,
            "category0": '',
            "vector": vector.astype(np.float32).tolist(),
        }

    # json_dict = json.dumps(temp_dict, ensure_ascii=False).encode('utf-8')
    # client.set(str(row["0"], json_dict))
    # print(row["0"], type(row["index"]), type(str(row["index"].astype(np.int_))), len(vector.astype(np.float32).tolist()))
    client.json().set("restaurant:"+str(int(row["index"])), Path.root_path(), temp_dict)