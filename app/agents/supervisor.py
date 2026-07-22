# app/agents/supervisor.py
import logging
from .agent import FaceRecognitionAgent

logger = logging.getLogger(__name__)

class SupervisorAgent:
    def __init__(self, face_app, collection):
        logger.info("Initializing Supervisor Agent...")
        self.face_recognition_agent = FaceRecognitionAgent(face_app, collection)

    def delegate(self, task_domain: str, operation: str, **kwargs):
        print(f"\n[1]  SUPERVISOR: Received request for Domain: '{task_domain}' | Operation: '{operation}'")
        if task_domain == "face_recognition":
            print("[2]  SUPERVISOR: Routing task to 🤖 FaceRecognitionAgent ->")
            return self.face_recognition_agent.handle_task(operation, **kwargs)
        else:
            raise ValueError(f"No agent found for domain: '{task_domain}'")