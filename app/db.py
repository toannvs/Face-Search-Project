from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

engine = create_engine("postgresql://admin:secret@postgres/face_db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()