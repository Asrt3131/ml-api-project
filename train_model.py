#!/usr/bin/env python3
"""
Train an Iris classification model and persist it for serving.

This script is production-grade: it logs structured metrics, performs
cross-validation, saves model + metadata atomically, and exposes a
CLI for reproducible training runs.

Author: A_srt343
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("models")
DEFAULT_MODEL_NAME = "model.pkl"
DEFAULT_METADATA_NAME = "model_metadata.json"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 100
DEFAULT_CV_FOLDS = 5

IRIS_CLASS_NAMES = ("setosa", "versicolor", "virginica")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    n_estimators: int = DEFAULT_N_ESTIMATORS
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    cv_folds: int = DEFAULT_CV_FOLDS
    output_dir: Path = DEFAULT_OUTPUT_DIR


@dataclass
class TrainingMetrics:
    """Serializable training metrics."""
    accuracy: float
    cv_mean: float
    cv_std: float
    cv_scores: list[float] = field(default_factory=list)
    classification_report: dict[str, Any] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)


@dataclass
class ModelMetadata:
    """Metadata persisted alongside the trained model."""
    model_type: str
    sklearn_version: str
    trained_at_utc: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    feature_names: list[str]
    target_names: list[str]


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load the Iris dataset and return features, targets, names."""
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    logger.info(
        "Dataset loaded: %d samples, %d features, %d classes",
        iris.data.shape[0],
        iris.data.shape[1],
        len(iris.target_names),
    )
    return iris.data, iris.target, list(iris.feature_names), list(iris.target_names)


def build_model(config: TrainingConfig) -> RandomForestClassifier:
    """Construct a RandomForestClassifier from config."""
    logger.info(
        "Building RandomForestClassifier (n_estimators=%d, random_state=%d)",
        config.n_estimators,
        config.random_state,
    )
    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=-1,
    )


def train_and_evaluate(
    config: TrainingConfig,
) -> tuple[RandomForestClassifier, TrainingMetrics, list[str], list[str]]:
    """Train the model and compute evaluation metrics."""
    X, y, feature_names, target_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    logger.info(
        "Train/test split: %d train, %d test", len(X_train), len(X_test)
    )

    model = build_model(config)
    logger.info("Fitting model...")
    model.fit(X_train, y_train)

    # Test-set evaluation
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    logger.info("Test accuracy: %.4f", accuracy)

    # Cross-validation (on full data, for stability estimate)
    logger.info("Running %d-fold cross-validation...", config.cv_folds)
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=config.cv_folds,
        scoring="accuracy",
        n_jobs=-1,
    )
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    logger.info("CV accuracy: %.4f ± %.4f", cv_mean, cv_std)

    # Rich report
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = TrainingMetrics(
        accuracy=accuracy,
        cv_mean=cv_mean,
        cv_std=cv_std,
        cv_scores=[float(s) for s in cv_scores],
        classification_report=report,
        confusion_matrix=cm,
    )
    return model, metrics, feature_names, target_names


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to `path` atomically (write-temp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with open(fd, "wb") as fh:
            fh.write(data)
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def save_model(model: RandomForestClassifier, path: Path) -> None:
    """Serialize the trained model to disk atomically."""
    logger.info("Saving model to %s", path)
    _atomic_write_bytes(path, pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def save_metadata(metadata: ModelMetadata, path: Path) -> None:
    """Persist model metadata as pretty-printed JSON atomically."""
    logger.info("Saving metadata to %s", path)
    payload = json.dumps(asdict(metadata), indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(path, payload)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Iris classification model and save it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help="Number of trees in the RandomForest.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Fraction of data reserved for testing (0 < test_size < 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory in which to save the model and metadata.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Sanity-check CLI arguments; raise SystemExit on failure."""
    if not (0.0 < args.test_size < 1.0):
        raise SystemExit(f"--test-size must be in (0, 1); got {args.test_size}")
    if args.n_estimators < 1:
        raise SystemExit(f"--n-estimators must be >= 1; got {args.n_estimators}")
    if args.cv_folds < 2:
        raise SystemExit(f"--cv-folds must be >= 2; got {args.cv_folds}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validate_args(args)

    config = TrainingConfig(
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        output_dir=args.output_dir,
    )

    try:
        model, metrics, feature_names, target_names = train_and_evaluate(config)

        # Persist artifacts
        model_path = config.output_dir / DEFAULT_MODEL_NAME
        metadata_path = config.output_dir / DEFAULT_METADATA_NAME

        save_model(model, model_path)

        metadata = ModelMetadata(
            model_type=type(model).__name__,
            sklearn_version=_sklearn_version(),
            trained_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            config=asdict(config),
            metrics=asdict(metrics),
            feature_names=feature_names,
            target_names=target_names,
        )
        # Path objects aren't JSON-serializable
        metadata.config["output_dir"] = str(metadata.config["output_dir"])
        save_metadata(metadata, metadata_path)

        logger.info("✅ Training complete.")
        logger.info("   Model:     %s", model_path)
        logger.info("   Metadata:  %s", metadata_path)
        logger.info("   Accuracy:  %.4f", metrics.accuracy)
        logger.info("   CV:        %.4f ± %.4f", metrics.cv_mean, metrics.cv_std)
        return 0

    except FileNotFoundError as exc:
        logger.error("Missing input: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed: %s", exc)
        return 1


def _sklearn_version() -> str:
    """Return the installed scikit-learn version."""
    import sklearn  # local import to keep module import lightweight
    return sklearn.__version__


if __name__ == "__main__":
    sys.exit(main())