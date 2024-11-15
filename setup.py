"""
Code samples for vector database quickstart pages:
    https://redis.io/docs/latest/develop/get-started/vector-database/
"""

import json
import time

import numpy as np
import pandas as pd
import requests
import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# URL = ("https://raw.githubusercontent.com/bsbodden/redis_vss_getting_started"
#        "/main/data/bikes.json"
#        )
# response = requests.get(URL, timeout=10)
# bikes = response.json()

# json.dumps(bikes[0], indent=2)

client = redis.Redis(host="localhost", port=6379, decode_responses=True)

res = client.ping()
# >>> True

# >>>
# {
#   "model": "Summit",
#   "brand": "nHill",
#   "price": 1200,
#   "type": "Mountain Bike",
#   "specs": {
#     "material": "alloy",
#     "weight": "11.3"
#   },
#   "description": "This budget mountain bike from nHill performs well..."
#   "description_embeddings": [
#     -0.538114607334137,
#     -0.49465855956077576,
#     -0.025176964700222015,
#     ...
#   ]
# }

df = pd.read_csv("hanam_mean_vectors_100000.csv",header=None)
# df

df_mean_vectors = pd.read_csv("hanam_mean_vectors_100000.csv",header=None)

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

dict_list = []

for i in range(len(df)):
    row = df.iloc[i]
    vector = df_mean_vectors.iloc[i]

    temp_dict = {
        "idx": row["0"],
        "name": row["name"],
        "addr": row["position"],
        "dist": row["total_distance"],
        "reqtime": row["total_time"],
        "category0": row["category3"],
        "vector": vector.to_numpy(dtype=np.float32),
    }

    json_dict = json.dumps(test_dict, ensure_ascii=False).encode('utf-8')
    client.set(str(row["0"], json_dict))