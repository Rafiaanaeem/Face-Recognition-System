import chromadb
from typing import List, Dict, Any
from app.config import settings
from fastapi import HTTPException, status

class ChromaService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        print("Database Path:", settings.CHROMA_PERSIST_DIR)
        print("Collection Name:", settings.COLLECTION_NAME)
        print("Total Embeddings:", self.collection.count())

    def upsert_face(self, face_id: str, embedding: List[float], metadata: Dict[str, Any]):
        try:
            self.collection.upsert(
                ids=[face_id],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database upsert error: {str(e)}"
            )
    def query_nearest(self, query_embedding: List[float]) -> Dict[str, Any]:
        try:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["metadatas", "distances"]
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database query error: {str(e)}"
            )
        
chroma_db = ChromaService()

