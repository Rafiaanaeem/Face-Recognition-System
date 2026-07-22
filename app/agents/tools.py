# app/agents/tools.py
import uuid
from app.config import settings 

class AddTool:
    def __init__(self, face_app, collection):
        self.app = face_app
        self.collection = collection

    def execute(self, image, person_name: str):
        faces = self.app.get(image)
        if len(faces) != 1:
            return {"success": False, "message": f"Expected 1 face, found {len(faces)}."}
        
        embedding = faces[0].normed_embedding.tolist()
        bbox = faces[0].bbox.astype(int).tolist()
        
        unique_id = f"{person_name.replace(' ', '_')}_{uuid.uuid4().hex[:6]}"

        self.collection.upsert(
            ids=[unique_id],
            embeddings=[embedding],
            metadatas=[{"person_name": person_name, "bbox": str(bbox)}]
        )
        print(f"✅ [AddTool] Enrolled '{person_name}' (ID: {unique_id})")
        return {"success": True, "message": f"Successfully enrolled {person_name}."}


class SearchTool:
    def __init__(self, face_app, collection, threshold=None):
        self.app = face_app
        self.collection = collection
        self.threshold = threshold or settings.COSINE_THRESHOLD

    def execute(self, image):
        faces = self.app.get(image)
        if len(faces) == 0:
            return {"success": False, "message": "No faces detected in image.", "results": []}

        print(f"📊 [SearchTool] Searching among {self.collection.count()} total vectors...")

        results_list = []
        for idx, face in enumerate(faces):
            query_embedding = face.normed_embedding.tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                include=["metadatas", "distances"]
            )
            
            if results and results.get("distances") and len(results["distances"][0]) > 0:
                distance = float(results["distances"][0][0])
                matched_person = results["metadatas"][0][0]["person_name"]
                
                print(f"🔍 [SearchTool] Face #{idx+1} -> Matched: '{matched_person}' | Distance: {distance:.4f}")
                similarity_score = max(0.0, 1.0 - distance)
                similarity_percent = round(similarity_score * 100, 1)

                # Check if it passes threshold
                if distance <= self.threshold:
                    match_name = matched_person
                else:
                    match_name = "Unknown"
            else:
                distance = 1.0
                similarity_percent = 0.0
                match_name = "Unknown"
                
            results_list.append({
               "match": match_name, 
               "cosine_distance": distance,
               "similarity_percent": similarity_percent, # 🟢 Added to response
               "bbox": face.bbox.astype(int).tolist()
})
            
        return {"success": True, "results": results_list}