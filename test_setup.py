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
from redis.commands.search.query import Query
# from sentence_transformers import SentenceTransformer


URL = ("https://raw.githubusercontent.com/bsbodden/redis_vss_getting_started"
       "/main/data/bikes.json"
       )
response = requests.get(URL, timeout=10)
bikes = response.json()

json.dumps(bikes[0], indent=2)

client = redis.Redis(host="localhost", port=6379, decode_responses=True)

res = client.ping()
# >>> True

pipeline = client.pipeline()
for i, bike in enumerate(bikes, start=1):
    redis_key = f"bikes:{i:03}"
    pipeline.json().set(redis_key, "$", bike)
res = pipeline.execute()
# >>> [True, True, True, True, True, True, True, True, True, True, True]

res = client.json().get("bikes:010", "$.model")
# >>> ['Summit']

keys = sorted(client.keys("bikes:*"))
# >>> ['bikes:001', 'bikes:002', ..., 'bikes:011']

descriptions = client.json().mget(keys, "$.description")
descriptions = [item for sublist in descriptions for item in sublist]
# embedder = SentenceTransformer("msmarco-distilbert-base-v4")
embeddings = embedder.encode(descriptions).astype(np.float32).tolist()
VECTOR_DIMENSION = len(embeddings[0])
# >>> 768

pipeline = client.pipeline()
for key, embedding in zip(keys, embeddings):
    pipeline.json().set(key, "$.description_embeddings", embedding)
pipeline.execute()
# >>> [True, True, True, True, True, True, True, True, True, True, True]

res = client.json().get("bikes:010")
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

schema = (
    TextField("$.model", no_stem=True, as_name="model"),
    TextField("$.brand", no_stem=True, as_name="brand"),
    NumericField("$.price", as_name="price"),
    TagField("$.type", as_name="type"),
    TextField("$.description", as_name="description"),
    VectorField(
        "$.description_embeddings",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIMENSION,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
)
definition = IndexDefinition(prefix=["bikes:"], index_type=IndexType.JSON)
res = client.ft("idx:bikes_vss").create_index(fields=schema, definition=definition)
# >>> 'OK'

info = client.ft("idx:bikes_vss").info()
num_docs = info["num_docs"]
indexing_failures = info["hash_indexing_failures"]
# print(f"{num_docs} documents indexed with {indexing_failures} failures")
# >>> 11 documents indexed with 0 failures