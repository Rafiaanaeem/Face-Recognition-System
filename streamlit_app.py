import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# Page Configuration
st.set_page_config(
    page_title="ArcFace AI Vision Dashboard",
    layout="wide"
)

# Backend API Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Aapka FastAPI server URL

# Helper Function: Draw Bounding Boxes on Image
def draw_results_on_image(image_bytes, matches):
    # Raw bytes ko OpenCV BGR Image matrix mein convert karna
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for match in matches:
        bbox = match["bounding_box"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        name = match["person_name"]
        is_known = match["is_known"]
        dist = match["cosine_distance"]

        # Color: Known ke liye Green (0, 255, 0), Unknown ke liye Red (0, 0, 255)
        color = (0, 255, 0) if is_known else (0, 0, 255)
        
        # Label text format
        label = f"{name} ({dist})" if is_known else f"Unknown ({dist})"

        # Bounding box draw karna
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Label Text Background Box
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = label_size
        cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        
        # Text render karna (White color)
        cv2.putText(
            img, label, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    # OpenCV BGR se PIL RGB conversion Streamlit display ke liye
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# Streamlit UI Header
st.title(" ArcFace AI Face Recognition System")
st.caption("Powered by FastAPI, InsightFace, ChromaDB & Streamlit")

# Sidebar for Server Status Check
with st.sidebar:
    st.header("⚙️ Server Status")
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        if response.status_code == 200:
            st.success("FastAPI Backend: ONLINE 🟢")
        else:
            st.warning("FastAPI Backend: UNREACHABLE 🟡")
    except Exception:
        st.error("FastAPI Backend: OFFLINE 🔴\n(Start main.py first)")

# Main Navigation Tabs
tab_search, tab_enroll = st.tabs(["🔍 Face Search & Recognition", "👤 Enroll New Person"])

# ---------------------------------------------------------
# TAB 1: SEARCH & RECOGNITION
# ---------------------------------------------------------
with tab_search:
    st.subheader("Upload Image(s) to Recognize Faces")
    
    uploaded_files = st.file_uploader(
        "Choose image files...", 
        type=["jpg", "jpeg", "png", "webp"], 
        accept_multiple_files=True,
        key="search_uploader"
    )

    if uploaded_files:
        if st.button(" Identify Face(s)", type="primary"):
            files_to_send = [
                ("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files
            ]

            with st.spinner("Processing AI Inference & Vector Search..."):
                try:
                    res = requests.post(f"{API_BASE_URL}/search", files=files_to_send)
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Processed {data.get('total_images_processed', 0)} image(s) successfully!")

                        for idx, result in enumerate(data.get("results", [])):
                            st.divider()
                            col1, col2 = st.columns([3, 2])

                            # Retrieve original image bytes
                            raw_bytes = uploaded_files[idx].getvalue()
                            matches = result.get("matches", [])

                            with col1:
                                st.markdown(f"**Image:** `{result['filename']}`")
                                # Draw boxes & display
                                annotated_img = draw_results_on_image(raw_bytes, matches)
                                st.image(annotated_img, use_container_width=True)

                            with col2:
                                st.markdown(f"⏱️ **Inference Time:** `{result['inference_time_sec']} sec`")
                                st.markdown(f"👥 **Faces Detected:** `{result['faces_detected']}`")

                                if matches:
                                    st.markdown("### Match Details:")
                                    for match in matches:
                                        status_icon = "✅" if match["is_known"] else "❓"
                                        st.info(
                                            f"**Face #{match['face_index']}**\n\n"
                                            f"• **Status:** {status_icon} {match['person_name']}\n\n"
                                            f"• **Cosine Distance:** `{match['cosine_distance']}`\n\n"
                                            f"• **Bounding Box:** `{match['bounding_box']}`"
                                        )
                                else:
                                    st.warning("No faces detected in this image.")

                    else:
                        st.error(f"Error {res.status_code}: {res.text}")

                except Exception as e:
                    st.error(f"Failed to connect to backend API: {e}")

# ---------------------------------------------------------
# TAB 2: ENROLL NEW PERSON
# ---------------------------------------------------------
with tab_enroll:
    st.subheader("Add a New Face to Database")

    person_name = st.text_input("Enter Person Name:", placeholder="e.g., Natasha Romanoff")
    enroll_files = st.file_uploader(
        "Upload Face Photos (1 or more):", 
        type=["jpg", "jpeg", "png", "webp"], 
        accept_multiple_files=True,
        key="enroll_uploader"
    )

    if st.button("📥 Save & Enroll Face(s)", type="primary"):
        if not person_name.strip():
            st.error("Please enter a person name!")
        elif not enroll_files:
            st.error("Please upload at least one image!")
        else:
            files_to_send = [
                ("files", (f.name, f.getvalue(), f.type)) for f in enroll_files
            ]

            with st.spinner("Extracting embeddings & updating ChromaDB..."):
                try:
                    # FIX 1: params ki jagah data use kiya hai (Form Data Validation ke liye)
                    res = requests.post(
                        f"{API_BASE_URL}/add", 
                        data={"person_name": person_name}, 
                        files=files_to_send
                    )

                    # FIX 2: 201 (Created) status code ko success list mein add kar diya
                    if res.status_code in [200, 201]:
                        data = res.json()
                        
                        st.success(f" {data.get('message', 'Enrollment completed!')}")
                        st.json(data)
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")

                except Exception as e:
                    st.error(f"Failed to connect to backend API: {e}")