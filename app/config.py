import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "ArcFace Production API"
    VERSION: str = "1.0.0"
    
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db/arcface_db")
    COLLECTION_NAME: str = "arcface_faces"
    
    COSINE_THRESHOLD: float = 0.60
    EMBEDDING_DIM: int = 512
    
    MODEL_NAME: str = "buffalo_l"
    DET_SIZE: tuple = (640, 640)
    
    class Config:
        case_sensitive = True

settings = Settings()