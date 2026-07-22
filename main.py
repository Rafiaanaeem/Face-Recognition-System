# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

# 🟢 1. Import your existing modules precisely
from app.api.routes import router as api_router
from app.services.chroma_service import chroma_db  # Aapka original DB
from app.models.model_loader import arcface       # Aapka InsightFace model
from app.agents.supervisor import SupervisorAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting FastAPI Server...")
    
    # 🟢 2. Pass the exact collection from chroma_service to your Supervisor
    # Yeh ensure karega ke Agent wahi DB parhay jahan aapka purana dataset para hai!
    app.state.supervisor = SupervisorAgent(
        face_app=arcface.app,              # Access the raw FaceAnalysis app
        collection=chroma_db.collection    # Access the exact Chroma collection
    )
    
    yield
    print("🛑 Shutting down FastAPI Server...")

app = FastAPI(title="ArcFace AI System", lifespan=lifespan)

# Register routes
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)