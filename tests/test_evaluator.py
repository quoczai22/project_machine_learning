"""Unit tests for shared evaluation metrics."""

import numpy as np
import pytest

from src.evaluator import classification_metrics, clustering_silhouette


def test_classification_metrics_returns_expected_values() -> None:
    """Metrics must use Churn=1 as the positive class."""
    results = classification_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))

    assert results["accuracy"] == pytest.approx(0.75)
    assert results["precision"] == pytest.approx(1.0)
    assert results["recall"] == pytest.approx(0.5)
    assert results["f1"] == pytest.approx(2 / 3)


def test_clustering_silhouette_rejects_single_cluster() -> None:
    """A one-cluster result is invalid for silhouette evaluation."""
    with pytest.raises(ValueError, match="at least two clusters"):
        clustering_silhouette(np.array([[0.0], [1.0]]), np.array([0, 0]))
