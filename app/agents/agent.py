# to import the two functions from tools.py
from .tools import AddTool, SearchTool
class FaceRecognitionAgent:
    def __init__(self, face_app, collection):
        # Initialize tools with necessary dependencies
        self.add_tool = AddTool(face_app, collection)
        self.search_tool = SearchTool(face_app, collection)


    def handle_task(self, operation: str, **kwargs):
        """
        operation: 'add' or 'search'
        kwargs: payload containing 'image' and optionally 'person_name'
        """
        print(f" AGENT (FaceRecognition): Processing operation: '{operation}'")
        if operation == "add":
            print("[4] 🤖 AGENT: Triggering 🛠️ AddTool...")
            return self.add_tool.execute(
                image=kwargs.get("image"), 
                person_name=kwargs.get("person_name")
            )
        elif operation == "search":
            print("[4] 🤖 AGENT: Triggering 🛠️ SearchTool...")
            return self.search_tool.execute(
                image=kwargs.get("image")
            )
        else:
            raise ValueError(f"FaceRecognitionAgent doesn't know operation: {operation}")