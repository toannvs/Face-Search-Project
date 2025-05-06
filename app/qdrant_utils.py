from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import os

# Initialize Qdrant client using the `QDRANT_HOST` environment variable
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def ensure_collection(company_id: str):
    """
    Ensure that a collection exists for the specified company.
    If not, create a new collection named `faces_<company_id>`.

    Args:
        company_id (str): Company ID (used for multi-tenancy).

    Returns:
        str: The name of the collection.
    """
    collection_name = f"faces_{company_id}"
    if not client.collection_exists(collection_name=collection_name):
        # Create a new collection with 512-dimensional vectors using cosine distance
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
    return collection_name

def insert_vector(company_id, vector, payload, point_id):
    """
    Insert a vector into the corresponding company's collection.

    Args:
        company_id (str): Company ID.
        vector (list[float]): The vector to be inserted.
        payload (dict): Metadata associated with the vector.
        point_id (str): Unique ID for the vector.

    Returns:
        None
    """
    # Ensure the collection exists before inserting the vector
    collection = ensure_collection(company_id)
    # Insert the vector into the collection
    client.upsert(
        collection_name=collection,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)]
    )

def search_vector(company_id, vector, top_k=3):
    """
    Search for similar vectors within the company's collection.

    Args:
        company_id (str): Company ID.
        vector (list[float]): The vector to search for.
        top_k (int, optional): Number of top results to return. Default is 1.

    Returns:
        list: List of matched results including vector and payload.
    """
    # Ensure the collection exists before searching
    collection = ensure_collection(company_id)
    # Perform the search and return results
    return client.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k,
        with_payload=True
    )
