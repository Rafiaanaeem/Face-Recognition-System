# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.services.chroma_service import chroma_db  
from app.models.model_loader import arcface      
from app.agents.supervisor import SupervisorAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting FastAPI Server...")
    
    app.state.supervisor = SupervisorAgent(
        face_app=arcface.app,              
        collection=chroma_db.collection    
    )
    yield
    print("Shutting down FastAPI Server...")

app = FastAPI(title="ArcFace AI System", lifespan=lifespan)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)