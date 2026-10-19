# 🤖 ML Prediction API — Iris Classification

> A production-grade Machine Learning API serving Iris flower classification predictions, built with FastAPI, scikit-learn, and Streamlit.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red.svg)](https://streamlit.io/)

---

## 📖 Overview

This project demonstrates a **complete ML service** — from training to serving:
- 🎓 **Training pipeline** — `train_model.py` builds and evaluates a `RandomForestClassifier`.
- ⚡ **FastAPI backend** — high-performance prediction API with validation, logging, and health probes.
- 🎨 **Streamlit frontend** — interactive UI with local model fallback.
- ✅ **Test suite** — 12 tests, 93% coverage.

---

## 🏗️ Architecture

```text
┌──────────────┐      HTTP       ┌──────────────┐     pickle    ┌──────────────┐
│  Streamlit   │ ──────────────▶ │   FastAPI    │ ────────────▶ │  model.pkl   │
│  (UI :8501)  │                 │ (API :8080)  │               │ RandomForest │
└──────────────┘                 └──────────────┘               └──────────────┘
```

---

## 🔗 Live Demo & Source Code

- 🚀 **Live Demo (Streamlit App):** [https://ml-api-project-8xte3fdm8ft2see9qzezu9.streamlit.app/](https://ml-api-project-8xte3fdm8ft2see9qzezu9.streamlit.app/)
- 📦 **GitHub Repository:** [https://github.com/Asrt3131/ml-api-project](https://github.com/Asrt3131/ml-api-project)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Asrt3131/ml-api-project.git
cd ml-api-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This produces `model.pkl`, ready for serving.

### 4. Run the FastAPI backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

API docs available at **http://localhost:8080/docs**

### 5. Run the Streamlit frontend

```bash
streamlit run app.py
```

UI available at **http://localhost:8501**

---

## 📡 API Reference

### `POST /predict`

**Request body:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**

```json
{
  "prediction": "setosa",
  "confidence": 1.0,
  "class_id": 0
}
```

### `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 🧪 Tests

Run the full test suite:

```bash
pytest --cov=. --cov-report=term-missing
```

Expected: **12 tests passing**, **~93% coverage**.

---

## 📂 Project Structure

```text
.
├── app.py                 # Streamlit frontend
├── main.py                # FastAPI application
├── train_model.py         # Training pipeline
├── model.pkl              # Trained RandomForest model
├── requirements.txt       # Python dependencies
├── tests/
│   └── test_api.py        # API test suite
└── README.md
```

---

## 🛠️ Tech Stack

| Layer     | Technology              |
|-----------|-------------------------|
| Backend   | FastAPI + Uvicorn       |
| ML        | scikit-learn            |
| Frontend  | Streamlit               |
| Testing   | pytest + pytest-cov     |

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

Built with ❤️ using the classic [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

---

> ⚠️ **Note:** I couldn't access your GitHub repository (the fetch was blocked), so the **Project Structure** and **test count** sections above are based on the original README. If your actual files differ (e.g., different filenames, extra modules), let me know and I'll adjust.
