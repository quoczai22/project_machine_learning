import numpy as np
import pandas as pd
import pytest

from src.clustering import FEATURES, evaluate, fit_selected


def make_data(n=160):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "tenure": rng.integers(0, 73, n),
        "MonthlyCharges": rng.uniform(18.25, 118.75, n),
        "service_count": rng.integers(0, 9, n),
    })


def test_evaluate_returns_both_algorithms():
    X, metrics = evaluate(make_data(), k_values=range(2, 5))
    assert X.shape == (160, 3)
    assert set(metrics.algorithm) == {"KMeans", "Hierarchical"}
    assert set(metrics.k) == {2, 3, 4}
    assert metrics.silhouette.between(-1, 1).all()


def test_fit_selected_has_expected_labels():
    result, comparison, X = fit_selected(make_data(), k=4)
    assert result.shape[0] == 160
    assert result.cluster_kmeans.nunique() == 4
    assert result.cluster_hierarchical.nunique() == 4
    assert comparison.shape == (2, 3)
