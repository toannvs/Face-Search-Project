from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .db import Base

class FaceRecord(Base):
    """
    The `face_records` table stores information about face vectors saved in Qdrant.

    Includes the following fields:
    - Auto-incrementing ID
    - Person's name
    - Company ID (for multi-tenancy support)
    - Corresponding ID in Qdrant
    - Saved image path
    - Record creation timestamp
    """

    __tablename__ = "face_records"

    id = Column(Integer, primary_key=True, index=True)  # Primary key, auto-incremented
    company_id = Column(String, index=True)             # Company ID (used for filtering by collection)
    name = Column(String)                               # Person's name
    qdrant_id = Column(String, unique=True)             # Corresponding UUID in Qdrant
    image_path = Column(String)                         # Path to the uploaded original image
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Record creation timestamp
