from pydantic import BaseModel

class FaceUpload(BaseModel):
    company_id: str
    name: str
