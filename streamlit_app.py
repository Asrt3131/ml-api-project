"""
ML Prediction Demo — Streamlit Frontend
=======================================
A production-grade UI for an Iris classification API (FastAPI backend),
with a resilient local-model fallback.

Author: A_srt343
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pickle
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.pkl")
DEFAULT_API_URL = "http://localhost:8080"
REQUEST_TIMEOUT = 3.0

FEATURE_LABELS = ("Sepal Length", "Sepal Width", "Petal Length", "Petal Width")
FEATURE_DEFAULTS = (5.1, 3.5, 1.4, 0.2)
FEATURE_RANGES = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0))

CLASS_NAMES = {
    0: "Setosa 🌸",
    1: "Versicolor 🌺",
    2: "Virginica 🌼",
}

st.set_page_config(
    page_title="ML Prediction Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PredictionResult:
    """Normalized prediction outcome from either API or local model."""
    label: int
    confidence: float
    source: str  # "api" | "local"


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_local_model() -> Any:
    """Load and cache the local pickle model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Local model not found at {MODEL_PATH}")
    with MODEL_PATH.open("rb") as fh:
        return pickle.load(fh)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def call_predict_api(api_url: str, features: list[float]) -> PredictionResult:
    """POST features to /predict and normalize the response."""
    resp = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"features": features},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    return PredictionResult(
        label=int(payload["prediction"]),
        confidence=float(payload["confidence"]),
        source="api",
    )


def call_health_api(api_url: str) -> dict[str, Any]:
    """GET /health and return the JSON payload."""
    resp = requests.get(
        f"{api_url.rstrip('/')}/health",
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def predict_local(features: list[float]) -> PredictionResult:
    """Run inference using the cached local model."""
    model = load_local_model()
    arr = np.asarray([features], dtype=float)
    label = int(model.predict(arr)[0])
    confidence = float(model.predict_proba(arr)[0][label])
    return PredictionResult(label=label, confidence=confidence, source="local")


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------
def render_header() -> None:
    st.title("🤖 ML Prediction Demo")
    st.caption(
        "FastAPI + Docker + Streamlit · Iris classification · "
        "Built with ❤️ by **A_srt343**"
    )


def render_sidebar_inputs() -> tuple[str, list[float]]:
    """Render sidebar controls and return (api_url, features)."""
    st.sidebar.header("⚙️ Configuration")
    api_url = st.sidebar.text_input(
        "API URL",
        value=DEFAULT_API_URL,
        help="Base URL of the FastAPI backend.",
    ).strip()

    st.sidebar.divider()
    st.sidebar.header("🌿 Input Features (Iris)")

    features: list[float] = []
    for label, default, (lo, hi) in zip(FEATURE_LABELS, FEATURE_DEFAULTS, FEATURE_RANGES):
        features.append(
            st.sidebar.number_input(
                label,
                min_value=lo,
                max_value=hi,
                value=default,
                step=0.1,
                format="%.2f",
            )
        )

    st.sidebar.divider()
    return api_url, features


def render_input_card(features: list[float]) -> None:
    st.subheader("📥 Input")
    st.json(
        {label: val for label, val in zip(FEATURE_LABELS, features)},
        expanded=True,
    )


def render_prediction_card(result: PredictionResult) -> None:
    """Beautifully render a prediction result."""
    class_name = CLASS_NAMES.get(result.label, f"Class {result.label}")
    source_badge = "🌐 API" if result.source == "api" else "💾 Local Fallback"

    st.success(f"**Prediction:** {class_name}")
    st.progress(min(max(result.confidence, 0.0), 1.0))
    st.metric(
        label="Confidence",
        value=f"{result.confidence * 100:.2f}%",
        help=f"Source: {source_badge}",
    )
    st.caption(f"Source: {source_badge}")

    if result.source == "api":
        st.balloons()


def handle_prediction(api_url: str, features: list[float]) -> None:
    """Orchestrate prediction with graceful fallbacks."""
    if not api_url:
        st.error("⚠️ API URL is empty. Please provide a valid URL.")
        return

    try:
        with st.spinner("Calling prediction API..."):
            result = call_predict_api(api_url, features)
        render_prediction_card(result)
        logger.info("API prediction succeeded: %s", result)

    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API unreachable. Falling back to local model...")
        _try_local_fallback(features)

    except requests.exceptions.Timeout:
        st.warning("⏱️ API timed out. Falling back to local model...")
        _try_local_fallback(features)

    except requests.exceptions.HTTPError as exc:
        st.error(
            f"🚫 API returned HTTP {exc.response.status_code}: "
            f"{exc.response.text}"
        )
        logger.exception("API HTTP error")

    except (KeyError, ValueError, TypeError) as exc:
        st.error(f"❌ Malformed API response: {exc}")
        logger.exception("Malformed API response")

    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Unexpected error: {exc}")
        logger.exception("Unexpected prediction error")


def _try_local_fallback(features: list[float]) -> None:
    """Attempt local prediction, reporting any failure gracefully."""
    try:
        with st.spinner("Running local inference..."):
            result = predict_local(features)
        render_prediction_card(result)
        logger.info("Local prediction succeeded: %s", result)
    except FileNotFoundError as exc:
        st.error(f"❌ Local model unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Local prediction failed: {exc}")
        logger.exception("Local prediction failed")


def render_health_check(api_url: str) -> None:
    """Sidebar health check with clean error handling."""
    st.sidebar.divider()
    st.sidebar.header("🩺 Diagnostics")

    if st.sidebar.button("Check API Health", use_container_width=True):
        if not api_url:
            st.sidebar.error("API URL is empty.")
            return
        try:
            with st.sidebar:
                with st.spinner("Pinging /health..."):
                    payload = call_health_api(api_url)
            st.sidebar.success("✅ API is healthy")
            st.sidebar.json(payload)
        except requests.exceptions.ConnectionError:
            st.sidebar.error("❌ API unreachable (connection error).")
        except requests.exceptions.Timeout:
            st.sidebar.error("⏱️ API health check timed out.")
        except requests.exceptions.HTTPError as exc:
            st.sidebar.error(f"🚫 Health check HTTP {exc.response.status_code}.")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"❌ Health check failed: {exc}")


def render_footer() -> None:
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; color:#888; font-size:0.9rem; line-height:1.6;">
            Built with ❤️ by <strong>A_srt343</strong><br>
            Instagram: <a href="https://instagram.com/project_srt343"
                         target="_blank"
                         style="color:#E1306C; text-decoration:none;">
                @project_srt343
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    render_header()
    api_url, features = render_sidebar_inputs()

    col_input, col_output = st.columns(2, gap="large")

    with col_input:
        render_input_card(features)

    with col_output:
        st.subheader("📤 Prediction")
        if st.button(
            "🚀 Predict",
            type="primary",
            use_container_width=True,
        ):
            handle_prediction(api_url, features)
        else:
            st.info("Set the features in the sidebar, then click **Predict**.")

    render_health_check(api_url)
    render_footer()


if __name__ == "__main__":
    main()