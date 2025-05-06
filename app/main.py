from fastapi import FastAPI, File, UploadFile, Form, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from . import face_utils, qdrant_utils, models, db
from .schemas import FaceUpload
from sqlalchemy.orm import Session
from uuid import uuid4
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize the FastAPI application with metadata for Swagger UI
app = FastAPI(
    title="Face Embedding Search API",
    description="Upload face images and search using multi-tenancy (per company).",
    version="1.0.0",
)

# Set up CORS middleware to allow frontend access from different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, can be configured during deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the database (create tables if they do not exist)
models.Base.metadata.create_all(bind=db.engine)

def get_db():
    """
    Dependency to get a database session.
    Ensures the session is closed after use.
    """
    db_session = db.SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()

@app.post(
    "/upload",
    summary="Upload face image",
    description="Upload an image, extract embedding, save to Qdrant and DB.",
    tags=["Face API"],
)
def upload_face(
    file: UploadFile = File(..., description="Image file containing a face"),
    name: str = Form(..., description="Person's name"),
    company_id: str = Form(..., description="Company ID (for multi-tenancy)"),
    db: Session = Depends(get_db)
):
    """
    API to upload a face image:
    - Extract face embedding from the image
    - Save the embedding to Qdrant (per company)
    - Save the information to PostgreSQL

    Args:
        file (UploadFile): Image containing the face
        name (str): Person's name
        company_id (str): Company ID (multi-tenancy partitioning)
        db (Session): Database connection

    Returns:
        dict: Processing result or error if no face is detected
    """
    # Extract the embedding vector from the image
    vector = face_utils.extract_embedding(file.file)
    if vector is None:
        return JSONResponse(status_code=400, content={"error": "No face detected"})

    # Reset the file pointer before saving
    file.file.seek(0)

    # Generate a unique ID for the vector
    image_path = face_utils.save_image(file, str(uuid4()))

    # Tạo ID duy nhất cho vector
    point_id = str(uuid4())

    # Save the vector to Qdrant
    qdrant_utils.insert_vector(
        company_id,
        vector.tolist(),
        {"name": name, "company_id": company_id},
        point_id
    )

    # Record the vector information in the database
    record = models.FaceRecord(
        name=name,
        company_id=company_id,
        qdrant_id=point_id,
        image_path=image_path
    )
    db.add(record)
    db.commit()

    return {"status": "ok", "qdrant_id": point_id, "image_path": image_path}

@app.post(
    "/search",
    summary="Search face",
    description="Search similar face in the same company only.",
    tags=["Face API"],
)
def search_face(
    file: UploadFile = File(..., description="Image file to search"),
    company_id: str = Form(..., description="Company ID to search within")
):
    """
    API to search for a similar face within the same company:
    - Extract the vector from the uploaded image
    - Query Qdrant only within the company's collection

    Args:
        file (UploadFile): Image containing the face to search
        company_id (str): Company ID (corresponding collection)

    Returns:
        dict: List of the most similar vectors (name, score, etc.), or error if no face is detected
    """
    # Extract the embedding from the image
    vector = face_utils.extract_embedding(file.file)
    if vector is None:
        return JSONResponse(status_code=400, content={"error": "No face found"})

    # Perform a query to find the closest vector in the company's collection
    results = qdrant_utils.search_vector(company_id, vector.tolist())
    return {"matches": results}
