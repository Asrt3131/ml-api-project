"""Model service: loading, inference, and lifecycle."""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)

IRIS_CLASS_NAMES: dict[int, str] = {0: "setosa", 1: "versicolor", 2: "virginica"}
FEATURE_NAMES: list[str] = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    path: Path
    loaded_at_utc: str
    sklearn_version: str | None


class ModelService:
    """Encapsulates the loaded model and all inference logic."""

    def __init__(self, model_path: Path) -> None:
        self._path = model_path
        self._bundle: ModelBundle | None = None

    # ---- lifecycle ----
    def load(self) -> None:
        logger.info("Loading model from %s", self._path)
        if not self._path.exists():
            raise FileNotFoundError(f"Model not found at {self._path}")
        with self._path.open("rb") as fh:
            model = pickle.load(fh)

        import sklearn
        self._bundle = ModelBundle(
            model=model,
            path=self._path,
            loaded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            sklearn_version=getattr(sklearn, "__version__", None),
        )
        logger.info("Model loaded: %s (sklearn %s)", type(model).__name__, self._bundle.sklearn_version)

    def unload(self) -> None:
        logger.info("Unloading model")
        self._bundle = None

    # ---- properties ----
    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    @property
    def bundle(self) -> ModelBundle:
        if self._bundle is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )
        return self._bundle

    # ---- inference ----
    def predict(self, features: list[float]) -> tuple[int, float]:
        arr = np.asarray([features], dtype=float)
        pred = int(self.bundle.model.predict(arr)[0])
        proba = float(self.bundle.model.predict_proba(arr)[0][pred])
        return pred, proba

    def predict_batch(self, batch: list[list[float]]) -> list[tuple[int, float]]:
        arr = np.asarray(batch, dtype=float)
        preds = self.bundle.model.predict(arr)
        probas = self.bundle.model.predict_proba(arr)
        return [(int(p), float(probas[i][int(p)])) for i, p in enumerate(preds)]


# Singleton — shared across the app
model_service = ModelService(settings.model_path)