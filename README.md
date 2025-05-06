# Face Search Project

This project provides a FastAPI-based service for face embedding extraction, storage, and search using Qdrant (vector database) and PostgreSQL (metadata storage). It supports multi-tenancy by company.

## Features

- Upload face images, extract embeddings, and store them in Qdrant
- Store face metadata in PostgreSQL
- Search for similar faces within a company
- RESTful API with OpenAPI (Swagger) documentation
- Dockerized for easy deployment

## Project Structure

```
.
├── app/
│   ├── db.py
│   ├── face_utils.py
│   ├── main.py
│   ├── models.py
│   ├── performance_test.py
│   ├── qdrant_utils.py
│   ├── schemas.py
├── images/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
```

## Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/)
- (Optional) Python 3.8+ if running locally without Docker

## Quick Start (Recommended)

1. **Cd to the project directory:**
   ```bash
   cd face-search-project
   ```

2. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```
   This will start:
   - FastAPI app (port 8000)
   - PostgreSQL database (port 5433)
   - Qdrant vector database (port 6333)

3. **Access the API documentation:**
   - Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

## API Endpoints

- `POST /upload`  
  Upload a face image, extract embedding, and store metadata.

- `POST /search`  
  Search for similar faces within a company.

See `/docs` for full API details.

## Environment Variables

- `QDRANT_HOST` (default: `qdrant`)
- `QDRANT_PORT` (default: `6333`)

These are set automatically by Docker Compose.

## Development (Without Docker)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start PostgreSQL and Qdrant manually** (see docker-compose.yml for configuration).

3. **Run the FastAPI app:**
   ```bash
   uvicorn app.main:app --reload
   ```

## Data Storage

- Uploaded images are saved in the images directory.
- Face embeddings are stored in Qdrant collections (one per company).
- Metadata is stored in PostgreSQL (`face_records` table).

## Notes

- The project uses [InsightFace](https://github.com/deepinsight/insightface) for face detection and embedding extraction.
- Make sure your machine has enough resources for running all services, especially for large-scale face data.
