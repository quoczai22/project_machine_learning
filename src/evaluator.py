"""Metrics shared by the classification and clustering workstreams."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, silhouette_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return reproducible binary-classification metrics for Churn=1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def clustering_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Calculate silhouette score after verifying that at least two clusters exist."""
    if len(set(labels)) < 2:
        raise ValueError("Silhouette score requires at least two clusters.")
    return float(silhouette_score(features, labels))
