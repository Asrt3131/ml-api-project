"""Pydantic schemas for request/response validation."""
from __future__ import annotations

from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field

from app.config import settings

# ---------------------------------------------------------------------------
# Reusable field types
# ---------------------------------------------------------------------------
FeatureVector = Annotated[
    list[float],
    Field(
        min_length=settings.expected_feature_count,
        max_length=settings.expected_feature_count,
        description="Exactly 4 numeric features (sepal/petal length & width).",
    ),
]


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
class PredictionInput(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),  # allow model_* anywhere in payloads
        json_schema_extra={"example": {"features": [5.1, 3.5, 1.4, 0.2]}},
    )
    features: FeatureVector


class BatchPredictionInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    samples: Annotated[
        list[PredictionInput],
        Field(min_length=1, max_length=settings.max_batch_size),
    ]


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------
class PredictionOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)


class BatchPredictionOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    count: int
    predictions: list[PredictionOutput]


class HealthOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # fixes model_loaded warning
    status: str
    model_loaded: bool


class ModelInfoOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # fixes model_type warning
    model_type: str
    path: str
    loaded_at_utc: str
    sklearn_version: str | None
    n_features: int
    feature_names: list[str]
    classes: list[str]


# ---------------------------------------------------------------------------
# Error envelope
# ---------------------------------------------------------------------------
class ErrorDetail(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    code: int
    message: str
    path: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    error: ErrorDetail