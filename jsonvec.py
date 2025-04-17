import redis
import json
import numpy as np
from redis.commands.search.field import VectorField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Initialize SentenceTransformerEmbeddings
embeddings = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Sample data
users = [
    {
        "id": "user_123",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30
    },
    {
        "id": "user_456",
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "age": 25
    }
]

# Generate 1024-dimensional embeddings for user names
vectors = [np.array(embeddings.embed_query(user["name"]), dtype=np.float32) for user in users]

# Index name
index_name = "user_idx"

def create_vector_index():
    """Create a Redis search index for JSON documents with vectors."""
    try:
        # Define schema
        schema = (
            TagField("$.id", as_name="id"),
            VectorField(
                "$.vector",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": 1024,  # For bge-large-en-v1.5
                    "DISTANCE_METRIC": "COSINE"
                },
                as_name="vector"
            )
        )

        # Create index
        redis_client.ft(index_name).create_index(
            fields=schema,
            definition=IndexDefinition(prefix=["user:"], index_type=IndexType.JSON)
        )
        print(f"Created index {index_name}")
    except redis.RedisError as e:
        if "Index already exists" not in str(e):
            raise e
        print(f"Index {index_name} already exists")

def store_json_and_vector(json_data, vector, key):
    """Store JSON data and vector in Redis."""
    try:
        # Store JSON data with vector
        data = json_data.copy()
        data["vector"] = vector.tobytes().hex()  # Store as hex for JSON compatibility
        redis_client.json().set(key, ".", data)
        print(f"Stored data for {key}")
    except redis.RedisError as e:
        print(f"Error storing data for {key}: {e}")

def retrieve_json(key):
    """Retrieve JSON data by key."""
    try:
        data = redis_client.json().get(key)
        if data:
            print(f"Retrieved JSON for {key}: {data}")
            return data
        else:
            print(f"No data found for {key}")
            return None
    except redis.RedisError as e:
        print(f"Error retrieving JSON for {key}: {e}")
        return None

def vector_similarity_search(query_text, top_k=2):
    """Perform vector similarity search using text query."""
    try:
        # Generate query vector
        query_vector = np.array(embeddings.embed_query(query_text), dtype=np.float32)

        # Convert query vector to hex
        query_vector_bytes = query_vector.tobytes().hex()

        # Build query
        query = Query(f"*=>[KNN {top_k} @vector $vec AS score]").sort_by("score").dialect(2)
        params = {"vec": bytes.fromhex(query_vector_bytes)}

        # Execute search
        results = redis_client.ft(index_name).search(query, query_params=params)

        # Process results
        for doc in results.docs:
            json_data = json.loads(doc.json)
            score = float(doc.score)
            print(f"Found document {doc.id}, Similarity: {1 - score:.4f}, Data: {json_data}")
        return results
    except redis.RedisError as e:
        print(f"Error in vector search: {e}")
        return None

def main():
    # Create index
    create_vector_index()

    # Store data
    for user, vector in zip(users, vectors):
        key = f"user:{user['id']}"
        store_json_and_vector(user, vector, key)

    # Retrieve JSON by key
    retrieve_json("user:user_123")

    # Perform vector similarity search
    query_text = "John Doe"
    print("\nPerforming vector similarity search for query: ", query_text)
    vector_similarity_search(query_text, top_k=2)

if __name__ == "__main__":
    try:
        main()
    finally:
        redis_client.close()
