# Face Recognition System using ArcFace, FastAPI, Multi-Agent Architecture & ChromaDB

## Overview

This project is a production-oriented Face Recognition System built using **ArcFace**, **FastAPI**, **ChromaDB**, **Streamlit**, and a **Multi-Agent Architecture**.

Instead of selecting a face recognition model based solely on research papers, multiple state-of-the-art models were benchmarked under identical conditions. The final model was selected using a **data-driven engineering approach** based on experimental results.

The system supports both **single-face** enrollment and **multiple-face** recognition and searching through a REST API and an interactive Streamlit interface.

---

# Project Objectives

- Build a production-quality Face Recognition System.
- Benchmark multiple face recognition models.
- Select the best model using experimental results.
- Store face embeddings in ChromaDB ( vector database)
- Support single face enrollment (may modify it for the multi-face enrollment)
- Support single and multiple face recognition.
- Implement a modular Multi-Agent architecture.

---

# Model Benchmarking

| Model | Accuracy | F1 Score | Inference Time | Model Loading Time |
|--------|----------|----------|----------------|--------------------|
| ArcFace | 100% | 100% | 1.14 sec | 12.29 sec |
| MagFace | 100% | 100% | 1.77 sec | 2.12 sec |
| AdaFace | 100% | 100% | 1.84 sec | 27.49 sec |
| FaceNet | 78.57% | 74.52% | 0.15 sec | 5.19 sec |

### Final Selected Model

**ArcFace** was selected because it achieved:

- 100% Accuracy
- 100% F1 Score
- Fast inference
- Low resource consumption
- Mature ecosystem
- Excellent community support
- Easy production deployment

---

# System Architecture

```text
                    Streamlit Frontend
                           │
                           ▼
                      FastAPI Router
                           │
                           ▼
                   Supervisor Agent
                           │
                           ▼
              Face Recognition Agent
                    │             │
                    ▼             ▼
              Add Face Tool   Search Face Tool
                    │             │
                    └──────┬──────┘
                           ▼
                    ArcFace (InsightFace)
                           │
                           ▼
                        ChromaDB
                           │
                           ▼
                  Recognition Results
```

---

# Features

- Multi-Agent Architecture
- Supervisor Agent
- Face Recognition Agent
- Add Face Tool
- Search Face Tool
- Face Detection (SCRFD)
- Face Alignment
- ArcFace Embedding Generation
- L2 Normalization
- ChromaDB Vector Database
- Cosine Similarity Search
- FastAPI REST API
- Streamlit Frontend
- Multiple-face Enrollment
- Multiple-face Recognition
- Benchmarking Framework
- CSV Result Export

---

# Technologies Used

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Computer Vision

- InsightFace
- ArcFace
- SCRFD
- OpenCV

### Machine Learning

- PyTorch
- NumPy

### Vector Database

- ChromaDB

### Frontend

- Streamlit

### Evaluation

- Pandas
- Scikit-learn
- tqdm

---

# Project Structure

```text
face_recognition/

├── app/
│   ├── agents/
│   │   ├── supervisor.py
│   │   ├── agent.py
│   │   └── tools.py
│   ├── api/
│   ├── config/
│   ├── models/
│   ├── services/
│   └── utils/
│
├── benchmark/
├── benchmark_results/
├── chroma_db/
├── dataset/
│   ├── enrollment/
│   └── testing/
│
├── streamlit_app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# API Endpoints

## POST `/add`

Enroll one person into the database.

**Input**

- Person Name
- One or more face images of th same person

**Response**

- Success Status
- Enrollment Details

---

## POST `/search`

Recognize one or more face images.

**Response**

- Person Name
- Similarity Score
- Bounding Box
- Known / Unknown Status

---

# Dataset

The benchmark dataset contains approximately **15–20 images** divided into:

```text
dataset/
├── enrollment/
└── testing/
```

Testing images include:

- Known people
- Unknown people
- Single-face images
- Multiple-face images
- Different lighting conditions
- Different poses

The same dataset was used for every benchmark to ensure a fair comparison.

---

# Vector Database

The system uses **ChromaDB** to store normalized ArcFace embeddings.

Each record contains:

- Unique ID
- Person Name
- Face Embedding
- Metadata

Recognition is performed using **Cosine Similarity Search**.

---

# Streamlit Frontend

The project includes a Streamlit application that communicates with the FastAPI backend.

Features:

- Enroll new identities
- Upload single or multiple images
- Search faces
- Display recognition results
- Display similarity scores
- Simple and user-friendly interface

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/face-recognition-system.git
```

Navigate to the project

```bash
cd face-recognition-system
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the FastAPI server

```bash
uvicorn main:app --reload
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

Run Streamlit

```bash
streamlit run streamlit_app.py
```

Default URL

```
http://localhost:8501
```

---

# Evaluation Metrics

The benchmark compared:

- Accuracy
- Precision
- Recall
- F1 Score
- Embedding Dimension
- Inference Time
- Model Loading Time
- CPU Usage
- RAM Usage
- GPU Utilization
- GPU VRAM Usage
- Model Size
- Database Search Time

---

# License

This project was developed for academic and educational purposes
By Rafia Naeem 