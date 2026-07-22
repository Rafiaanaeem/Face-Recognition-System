# app/api/routes.py
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException
import cv2
import numpy as np
import time

router = APIRouter()

async def read_image_bytes(upload_file: UploadFile):
    contents = await upload_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(
            status_code=400, 
            detail=f"Could not decode image '{upload_file.filename}'. Make sure it is a valid image file."
        )
    return img


@router.post("/search")
async def search_faces(request: Request, files: list[UploadFile] = File(...)):
    supervisor = request.app.state.supervisor
    processed_results = []
    
    for file in files:
        start_time = time.time()
        
        # Safe async read
        image = await read_image_bytes(file)
        
        # Supervisor call
        tool_response = supervisor.delegate(
            task_domain="face_recognition",
            operation="search",
            image=image
        )
        
        inference_time = round(time.time() - start_time, 4)
        
        matches_formatted = []
        raw_matches = tool_response.get("results", [])
        
        for idx, match in enumerate(raw_matches):
            is_known = match.get("match", "Unknown") != "Unknown"
            bbox = match.get("bbox", [0, 0, 0, 0])
            
            matches_formatted.append({
                "face_index": idx + 1,
                "person_name": match.get("match", "Unknown"),
                "is_known": is_known,
                "similarity_percent": match.get("similarity_percent", 0.0), # 🟢 Added
                "cosine_distance": round(float(match.get("cosine_distance", 1.0)), 4),
                "bounding_box": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3])
                }
            })
            
        processed_results.append({
            "filename": file.filename,
            "inference_time_sec": inference_time,
            "faces_detected": len(matches_formatted),
            "matches": matches_formatted
        })
        
    return {
        "total_images_processed": len(files),
        "results": processed_results
    }


@router.post("/add")
async def add_faces(
    request: Request, 
    person_name: str = Form(...), 
    files: list[UploadFile] = File(...)
):
    supervisor = request.app.state.supervisor
    enrollment_details = []
    
    for file in files:
        # Safe async read
        image = await read_image_bytes(file)
        
        # Supervisor call
        tool_response = supervisor.delegate(
            task_domain="face_recognition",
            operation="add",
            image=image,
            person_name=person_name
        )
        enrollment_details.append(tool_response)
        
    return {
        "success": True,
        "message": f"Successfully enrolled {person_name} with {len(files)} image(s).",
        "details": enrollment_details
    }