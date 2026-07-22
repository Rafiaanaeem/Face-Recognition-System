import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(
    page_title="ArcFace AI Vision Dashboard",
    layout="wide"
)

# Backend API Configuration
API_BASE_URL = "http://127.0.0.1:8000"  


def draw_results_on_image(image_bytes, matches):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    for match in matches:
        bbox = match["bounding_box"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        name = match["person_name"]
        is_known = match["is_known"]

        if "similarity_percent" in match:
            sim_pct = match["similarity_percent"]
        else:
            dist = match.get("cosine_distance", 1.0)
            sim_pct = round(max(0.0, 1.0 - dist) * 100, 1)

        color = (0, 255, 0) if is_known else (0, 0, 255)

        label = f"{name} ({sim_pct}%)" if is_known else f"Unknown ({sim_pct}%)"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        label_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = label_size

        label_y1 = max(y1 - text_h - 10, 0)
        label_y2 = y1 if y1 - text_h - 10 >= 0 else y1 + text_h + 10
        text_y = y1 - 5 if y1 - text_h - 10 >= 0 else y1 + text_h + 5

        cv2.rectangle(img, (x1, label_y1), (x1 + text_w + 10, label_y2), color, -1)

        cv2.putText(
            img, label, (x1 + 5, text_y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# Streamlit UI Header
st.title(" ArcFace AI Face Recognition System")
st.caption("Powered by FastAPI, InsightFace, ChromaDB & Streamlit")

with st.sidebar:
    st.header(" Server Status")
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=2)
        if response.status_code == 200:
            st.success("FastAPI Backend: ONLINE 🟢")
        else:
            st.warning("FastAPI Backend: UNREACHABLE 🟡")
    except Exception:
        st.error("FastAPI Backend: OFFLINE 🔴\n(Start main.py first)")

tab_search, tab_enroll = st.tabs(["🔍 Face Search & Recognition", "👤 Enroll New Person"])
# tab 1
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
                                # Draw annotated image with name & percentage
                                annotated_img = draw_results_on_image(raw_bytes, matches)
                                st.image(annotated_img, use_container_width=True)

                            with col2:
                                st.markdown(f"⏱️ **Inference Time:** `{result['inference_time_sec']} sec`")
                                st.markdown(f"👥 **Faces Detected:** `{result['faces_detected']}`")

                                if matches:
                                    st.markdown("### Match Details:")
                                    for match in matches:
                                        status_icon = "✅" if match["is_known"] else "❓"

                                        # Extract Similarity Percent
                                        if "similarity_percent" in match:
                                            sim_pct = match["similarity_percent"]
                                        else:
                                            dist = match.get("cosine_distance", 1.0)
                                            sim_pct = round(max(0.0, 1.0 - dist) * 100, 1)

                                        st.info(
                                            f"**Face #{match['face_index']}**\n\n"
                                            f"• **Status:** {status_icon} {match['person_name']}\n\n"
                                            f"• **Similarity Score:** `{sim_pct}%`\n\n"
                                            f"• **Cosine Distance:** `{match['cosine_distance']}`\n\n"
                                            f"• **Bounding Box:** `{match['bounding_box']}`"
                                        )
                                else:
                                    st.warning("No faces detected in this image.")

                    else:
                        st.error(f"Error {res.status_code}: {res.text}")

                except Exception as e:
                    st.error(f"Failed to connect to backend API: {e}")

# TAB 2: ENROLL NEW PERSON
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
                    res = requests.post(
                        f"{API_BASE_URL}/add",
                        data={"person_name": person_name},
                        files=files_to_send
                    )

                    if res.status_code in [200, 201]:
                        data = res.json()
                        st.success(f" {data.get('message', 'Enrollment completed!')}")
                        st.json(data)
                    else:
                        st.error(f"Error {res.status_code}: {res.text}")

                except Exception as e:
                    st.error(f"Failed to connect to backend API: {e}")