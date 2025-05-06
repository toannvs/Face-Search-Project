import numpy as np
import cv2
from insightface.app import FaceAnalysis
from uuid import uuid4
import os

# Initialize the InsightFace model
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use CPU; adjust if GPU is available
face_app.prepare(ctx_id=0, det_size=(640, 640))
def extract_embedding(file):
    """
    Extracts the face embedding from an image file.

    Args:
        file: File-like object containing the image.

    Returns:
        np.ndarray: Embedding vector or None if no face is detected.
    """
    # Read the image from the file
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Validate the image
    if img is None or len(img.shape) != 3:
        print("Invalid image file")
        return None

    # Detect faces and extract embeddings
    faces = face_app.get(img)
    if not faces:
        print("No faces detected")
        return None

    return faces[0].embedding

def save_image(file, name):
    """
    Save the image file to the `images/` directory with the specified name.

    Args:
        file: Uploaded image file (UploadFile from FastAPI).
        name (str): File name without extension.

    Returns:
        str: The file path of the saved image.
    """
    # Define the target image path
    path = f"images/{name}.jpg"
    # Create the `images` directory if it does not exist
    os.makedirs("images", exist_ok=True)
    # Write the image content to the destination file
    with open(path, "wb") as f:
        f.write(file.file.read())
    # Return the saved image path
    return path
