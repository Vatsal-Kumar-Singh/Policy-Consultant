from os import error
import redis
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# r = redis.Redis(
#     host=st.secrets["REDIS_HOST"],
#     port=10715,
#     decode_responses=True,
#     username="default",
#     password=st.secrets["REDIS_PASSWORD"],
# )


r = redis.Redis(
    host='redis-11772.c15.us-east-1-4.ec2.redns.redis-cloud.com',
    port=11772,
    decode_responses=True,
    username="default",
    password="zvJOvJ3bRMnXnSZav40HKW0Qzvl2KZvz",
)



def get_embeddings_model():
    global embeddings_model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings_model

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def normalize_query(user_query):
    return user_query.lower().replace(" ", "")

def check_redis_cache(user_query, threshold = 0.60):

    try:
        all_keys = r.keys('*')
        if not all_keys:
            print("Redis cache is empty. Pinging Vector DB instead.")            
            return None
        
        else:
            normalized_user_query = normalize_query(user_query)
            embeddings_model = get_embeddings_model()
            embedded_user_query = embeddings_model.embed_query(normalized_user_query)

            for key in all_keys:
                data = json.loads(r.get(key))
                cached_vector = np.array(data["vector"])
                similarity_score = cosine_similarity(np.array(embedded_user_query), cached_vector)
                if( similarity_score >= threshold):
                    print(f"Found similar query in cache, Skipping DB call. Similarity score: {similarity_score}")

                    best_match = data["answer"]
                    return best_match
                
            print("No similar query found in cache. Pinging Vector DB instead.")

    except redis.ConnectionError:
        print("Could not connect to Redis. Pinging Vector DB instead.")
        return None
    
def cache_query_answer(user_query, answer, ttl_seconds=300):
    global embeddings_model
    try:
        normalized_user_query = normalize_query(user_query)
        embeddings_model = get_embeddings_model()
        embedded_user_query = embeddings_model.embed_query(normalized_user_query)
        key = user_query
        r.set(key, json.dumps({"answer": answer, "vector": embedded_user_query}), ex=300)
        print("Successfully cached user query. TTL: 300 seconds")
    except Exception as e:
        print("Error while caching user query:", e)


