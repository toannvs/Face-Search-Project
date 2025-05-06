import time
import psutil
import numpy as np
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy.orm import Session
from . import db, models  # Adjust import if needed

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://qdrant:6333")

# Get database session
def get_db():
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

def generate_data(num_faces, vector_size=128):
    """Generate synthetic face embeddings and payloads."""
    vectors = np.random.rand(num_faces, vector_size).tolist()
    payloads = [{"name": f"Person_{i}"} for i in range(num_faces)]
    points = [PointStruct(id=str(uuid4()), vector=vectors[i], payload=payloads[i]) for i in range(num_faces)]
    return vectors, payloads, points

def insert_qdrant_data(collection_name, points, batch_size=1000):
    """Insert face embeddings into Qdrant in batches."""
    for i in range(0, len(points), batch_size):
        qdrant_client.upsert(collection_name=collection_name, points=points[i:i + batch_size])

def insert_postgres_data(db: Session, payloads, qdrant_ids):
    """Insert face metadata into PostgreSQL."""
    records = [
        models.FaceRecord(
            name=payloads[i]["name"],
            company_id="1",
            qdrant_id=qdrant_ids[i],
            image_path=f"/dummy/path/{qdrant_ids[i]}.jpg"
        )
        for i in range(len(payloads))
    ]
    db.bulk_save_objects(records)
    db.commit()

def search_qdrant(collection_name, query_vector, top_k=3):
    """Search similar face vectors in Qdrant."""
    return qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True
    )

def measure_performance(num_faces, collection_name="faces_1"):
    """Test performance for inserting and searching data in both Qdrant and PostgreSQL."""
    vectors, payloads, points = generate_data(num_faces)
    qdrant_ids = [point.id for point in points]

    # Insert into Qdrant
    start_qdrant_insert = time.time()
    insert_qdrant_data(collection_name, points)
    qdrant_insert_time = time.time() - start_qdrant_insert

    # Insert into PostgreSQL
    db_gen = get_db()
    db_session = next(db_gen)
    start_pg_insert = time.time()
    insert_postgres_data(db_session, payloads, qdrant_ids)
    pg_insert_time = time.time() - start_pg_insert

    # Search in Qdrant
    query_vector = np.random.rand(128).tolist()
    start_qdrant_search = time.time()
    search_qdrant(collection_name, query_vector)
    qdrant_search_time = time.time() - start_qdrant_search

    db_gen.close()

    # Memory usage
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024)

    return {
        "num_faces": num_faces,
        "qdrant_insert_time": qdrant_insert_time,
        "postgres_insert_time": pg_insert_time,
        "qdrant_search_time": qdrant_search_time,
        "postgres_search_time": pg_search_time,
        "memory_usage_mb": memory_usage
    }

# Run tests
for num in [1000, 10000, 100000, 1000000]:
    result = measure_performance(num)
    print(f"Faces: {result['num_faces']}, "
          f"Qdrant Insert: {result['qdrant_insert_time']:.2f}s, "
          f"Postgres Insert: {result['postgres_insert_time']:.2f}s, "
          f"Qdrant Search: {result['qdrant_search_time']:.2f}s, "
          f"Postgres Search: {result['postgres_search_time']:.2f}s, "
          f"Memory: {result['memory_usage_mb']:.2f}MB")
