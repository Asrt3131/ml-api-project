\# 🤖 ML Prediction API — Iris Classification



> A production-grade Machine Learning API serving Iris flower classification predictions, built with FastAPI, scikit-learn, and Streamlit.



\[!\[Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

\[!\[FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)

\[!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-1.40-red.svg)](https://streamlit.io/)



\---



\## 📖 Overview



This project demonstrates a \*\*complete ML service\*\* — from training to serving:

\- 🎓 \*\*Training pipeline\*\* — `train\_model.py` builds and evaluates a `RandomForestClassifier`.

\- ⚡ \*\*FastAPI backend\*\* — high-performance prediction API with validation, logging, and health probes.

\- 🎨 \*\*Streamlit frontend\*\* — interactive UI with local model fallback.

\- ✅ \*\*Test suite\*\* — 12 tests, 93% coverage.



\---



\## 🏗️ Architecture



```text

┌──────────────┐      HTTP       ┌──────────────┐     pickle    ┌──────────────┐

│  Streamlit   │ ──────────────▶ │   FastAPI    │ ────────────▶ │  model.pkl   │

│  (UI :8501)  │                 │ (API :8080)  │               │ RandomForest │

└──────────────┘                 └──────────────┘               └──────────────┘

```



\---



\## 🔗 Live Demo \& Source Code



\- 🚀 \*\*Live Demo (Streamlit App):\*\* \[https://YOUR-APP-URL.streamlit.app](https://YOUR-APP-URL.streamlit.app)

\- 📦 \*\*GitHub Repository:\*\* \[https://github.com/YOUR-USERNAME/YOUR-REPO](https://github.com/YOUR-USERNAME/YOUR-REPO)



> Replace the placeholders above (`YOUR-APP-URL` and `YOUR-USERNAME/YOUR-REPO`) with your actual deployment and repository links.



\---



\## 🚀 Quick Start



\### 1. Clone the repository



```bash

git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git

cd YOUR-REPO

```



\### 2. Install dependencies



```bash

pip install -r requirements.txt

```



\### 3. Train the model



```bash

python train\_model.py

```



This produces `model.pkl`, ready for serving.



\### 4. Run the FastAPI backend



```bash

uvicorn main:app --host 0.0.0.0 --port 8080 --reload

```



API docs available at \*\*http://localhost:8080/docs\*\*



\### 5. Run the Streamlit frontend



```bash

streamlit run app.py

```



UI available at \*\*http://localhost:8501\*\*



\---



\## 📡 API Reference



\### `POST /predict`



\*\*Request body:\*\*



```json

{

&#x20; "sepal\_length": 5.1,

&#x20; "sepal\_width": 3.5,

&#x20; "petal\_length": 1.4,

&#x20; "petal\_width": 0.2

}

```



\*\*Response:\*\*



```json

{

&#x20; "prediction": "setosa",

&#x20; "confidence": 1.0,

&#x20; "class\_id": 0

}

```



\### `GET /health`



\*\*Response:\*\*



```json

{

&#x20; "status": "healthy",

&#x20; "model\_loaded": true

}

```



\---



\## 🧪 Tests



Run the full test suite:



```bash

pytest --cov=. --cov-report=term-missing

```



Expected: \*\*12 tests passing\*\*, \*\*\~93% coverage\*\*.



\---



\## 📂 Project Structure



```text

.

├── app.py                 # Streamlit frontend

├── main.py                # FastAPI application

├── train\_model.py         # Training pipeline

├── model.pkl              # Trained RandomForest model

├── requirements.txt       # Python dependencies

├── tests/

│   └── test\_api.py        # API test suite

└── README.md

```



\---



\## 🛠️ Tech Stack



| Layer     | Technology              |

|-----------|-------------------------|

| Backend   | FastAPI + Uvicorn       |

| ML        | scikit-learn            |

| Frontend  | Streamlit               |

| Testing   | pytest + pytest-cov     |



\---



\## 📄 License



MIT — see \[LICENSE](LICENSE) for details.



\---



\## 🙌 Acknowledgements



Built with ❤️ using the classic \[Iris dataset](https://scikit-learn.org/stable/auto\_examples/datasets/plot\_iris\_dataset.html).

