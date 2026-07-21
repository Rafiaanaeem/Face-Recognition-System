# Face Recognition System using ArcFace, FastAPI & ChromaDB

## Overview

This project is a production-oriented Face Recognition System built using **ArcFace**, **FastAPI**, and **ChromaDB**. The primary objective is to provide accurate face enrollment and identification while following modern software engineering and machine learning best practices.

Unlike projects that simply adopt a model based on research papers, this system follows a **data-driven engineering approach**. Multiple state-of-the-art face recognition models were benchmarked under identical experimental conditions, and the final model was selected based on experimental performance.


# Project Objectives

* Build a production-quality Face Recognition API.
* Compare multiple state-of-the-art face recognition models.
* Select the final model using benchmark results instead of assumptions.
* Store facial embeddings inside a Vector Database.
* Support both single-face and multiple-face enrollment.
* Support both single-face and multiple-face recognition.
* Build a modular and scalable FastAPI application.


# Model Benchmarking

The following face recognition models were benchmarked using the same dataset and evaluation methodology.

| Model   | Accuracy | F1 Score | Inference Time | Model Loading Time |
| ------- | -------- | -------- | -------------- | ------------------ |
| ArcFace | 100%     | 100%     | 1.14 sec       | 12.29 sec          |
| MagFace | 100%     | 100%     | 1.77 sec       | 2.12 sec           |
| AdaFace | 100%     | 100%     | 1.84 sec       | 27.49 sec          |
| FaceNet | 78.57%   | 74.52%   | 0.15 sec       | 5.19 sec           |

### Final Selected Model

**ArcFace** was selected as the production model because it achieved:

* 100% Recognition Accuracy
* 100% F1 Score
* Stable inference performance
* Moderate GPU memory usage
* Excellent community support
* Mature production ecosystem
* Seamless integration through the official InsightFace implementation

The selection was based on experimental benchmarking rather than research claims.

---

# System Architecture

```text
Client
   │
   ▼
FastAPI
   │
   ▼
POST /add
POST /search
   │
   ▼
ArcFace Pipeline
   │
   ├── Face Detection (SCRFD)
   ├── Face Alignment
   ├── Face Embedding Generation
   └── L2 Normalization
   │
   ▼
ChromaDB
   │
   ▼
Cosine Similarity Search
   │
   ▼
Recognition Results
```


# Features

* Face Detection using SCRFD
* Face Alignment
* ArcFace Embedding Generation
* L2 Normalized Embeddings
* ChromaDB Vector Database
* Cosine Similarity Search
* FastAPI REST API
* Multiple-face Enrollment
* Multiple-face Recognition
* Benchmarking Framework
* CSV Result Export
* Modular Project Structure

---

# Technologies Used

## Backend

* FastAPI
* Uvicorn
* Pydantic

## Computer Vision

* InsightFace
* ArcFace
* Adaface
* Magface
* Facenet
* SCRFD
* OpenCV

## Machine Learning

* NumPy
* PyTorch

## Vector Database

* ChromaDB

## Evaluation

* Scikit-learn
* Pandas
* tqdm

---

# Project Structure

```text
face_recognition/

│
├── app/
│   ├── api/
│   ├── config/
│   ├── database/
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
├── main.py
├── config.py
├── stramlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

Streamlit Frontend

The project includes a Streamlit-based user interface that communicates with the FastAPI backend.

The frontend provides an easy way to interact with the Face Recognition API without using Swagger or Postman.

Features
Enroll new identities
Upload one or multiple face images
Search one or multiple face images
Display recognition results
Show similarity scores
Display uploaded images
User-friendly interface
---

# API Endpoints

## POST /add

Enroll one or more face images into the vector database.

### Request

* Person Name
* One or more images
```

### Response

* Success Status
* Total Faces Enrolled
* Enrollment Details

---

## POST /search

Recognize one or more uploaded face images.

```

### Response

* Person Name
* Similarity Score
* Bounding Box
* Recognition Status

---

# Dataset

The benchmark dataset contains approximately 15–20 images divided into two subsets.

```
dataset/

├── enrollment/
└── testing/
```

Testing images include:

* Single-face images
* Multiple-face images
* Known identities
* Unknown identities
* Different poses
* Different lighting conditions

The same dataset was used for all benchmarked models to ensure a fair comparison.

---

# Vector Database

The system uses ChromaDB for storing face embeddings.

Each record contains:

* Unique ID
* Person Name
* Face Embedding
* Metadata

Cosine Similarity is used during retrieval.

---

# Evaluation Metrics

The following metrics were measured during benchmarking:

* Accuracy
* Precision
* Recall
* F1 Score
* Embedding Dimension
* Inference Time
* Model Loading Time
* CPU Usage
* RAM Usage
* GPU Utilization
* GPU VRAM Usage
* Model Size
* Database Search Time

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/face-recognition-system.git
```

Navigate into the project:

```bash
cd face-recognition-system
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

```bash
uvicorn main:app --reload


```
Swagger Documentation:

http://127.0.0.1:8000/docs
```

Run the Streamlit Frontend

Open another terminal.
Activate the virtual environment again.

Run:
streamlit run streamlit_app.py

The Streamlit application will open automatically in your browser.

Default URL:
http://localhost:8501
