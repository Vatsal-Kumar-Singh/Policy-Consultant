import os
import redis
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Prefer environment variables or Streamlit secrets for Redis configuration.
# Fallback to hard-coded values only if no env/secrets are provided.
def _make_redis_client():
    host = os.environ.get("REDIS_HOST")
    port = os.environ.get("REDIS_PORT")
    password = os.environ.get("REDIS_PASSWORD")

    # If running inside Streamlit with secrets configured, prefer them.
    try:
        if 'st' in globals() and hasattr(st, 'secrets') and st.secrets.get('REDIS_HOST'):
            host = host or st.secrets.get('REDIS_HOST')
            port = port or st.secrets.get('REDIS_PORT')
            password = password or st.secrets.get('REDIS_PASSWORD')
    except Exception:
        # don't fail if st isn't fully available at import time
        pass

    if host and port:
        try:
            return redis.Redis(host=host, port=int(port), decode_responses=True, password=password, username="default")
        except Exception:
            pass

    # Last resort: try to use an in-repo default (kept for compatibility but not recommended)
    try:
        return redis.Redis(
            host='redis-11772.c15.us-east-1-4.ec2.redns.redis-cloud.com',
            port=11772,
            decode_responses=True,
            username="default",
            password="zvJOvJ3bRMnXnSZav40HKW0Qzvl2KZvz",
        )
    except Exception:
        # If creating the client fails, return None and allow callers to handle missing Redis gracefully
        return None


r = _make_redis_client()



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
        if r is None:
            print("Redis client not configured; skipping cache check.")
            return None

        all_keys = r.keys('*')
        if not all_keys:
            print("Redis cache is empty. Pinging Vector DB instead.")
            return None

        normalized_user_query = normalize_query(user_query)
        embeddings_model = get_embeddings_model()
        embedded_user_query = embeddings_model.embed_query(normalized_user_query)

        for key in all_keys:
            raw = r.get(key)
            if not raw:
                continue
            data = json.loads(raw)
            cached_vector = np.array(data.get("vector", []))
            if cached_vector.size == 0:
                continue
            similarity_score = cosine_similarity(np.array(embedded_user_query), cached_vector)
            if similarity_score >= threshold:
                print(f"Found similar query in cache, Skipping DB call. Similarity score: {similarity_score}")
                best_match = data.get("answer")
                return best_match

        print("No similar query found in cache. Pinging Vector DB instead.")

    except redis.ConnectionError:
        print("Could not connect to Redis. Pinging Vector DB instead.")
        return None
    
def cache_query_answer(user_query, answer, ttl_seconds=300):
    global embeddings_model
    try:
        if r is None:
            print('Redis not configured; cannot cache answer.')
            return

        normalized_user_query = normalize_query(user_query)
        embeddings_model = get_embeddings_model()
        embedded_user_query = embeddings_model.embed_query(normalized_user_query)
        key = user_query
        r.set(key, json.dumps({"answer": answer, "vector": embedded_user_query}), ex=ttl_seconds)
        print(f"Successfully cached user query. TTL: {ttl_seconds} seconds")
    except Exception as e:
        print("Error while caching user query:", e)


