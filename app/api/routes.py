from fastapi import APIRouter, File, UploadFile, Form, Query, status
from typing import List, Any, Dict, Annotated
from pydantic import BaseModel
from app.services.face_service import FaceService

router = APIRouter()

class AddResponse(BaseModel):
    success: bool
    message: str
    total_faces_enrolled: int
    details: List[Dict[str, Any]]

class SearchResponse(BaseModel):
    success: bool
    total_images_processed: int
    results: List[Dict[str, Any]]

# --- API Endpoints ---
@router.post("/add", response_model=AddResponse, status_code=status.HTTP_201_CREATED)
async def add_faces(
    person_name: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)]
):
    """Enroll single face for an identity."""
    return await FaceService.enroll(person_name, files)


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_faces(
    files: List[UploadFile] = File(...)
):
    """Search single or multiple query images against the database."""
    return await FaceService.search(files)