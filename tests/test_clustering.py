import numpy as np
import pandas as pd

from src.clustering import (
    build_service_count,
    evaluate_cluster_candidates,
    evaluate_dbscan_candidates,
    scale_cluster_features,
    select_cluster_features,
)


def make_telco_sample():
    return pd.DataFrame(
        {
            "tenure": [1, 12, 60, 24, 48, 3],
            "MonthlyCharges": [30, 45, 100, 80, 95, 25],
            "PhoneService": ["No", "Yes", "Yes", "Yes", "Yes", "No"],
            "MultipleLines": ["No phone service", "No", "Yes", "No", "Yes", "No phone service"],
            "OnlineSecurity": ["No", "No", "Yes", "Yes", "No", "No"],
            "OnlineBackup": ["No", "Yes", "Yes", "No", "Yes", "No"],
            "DeviceProtection": ["No", "No", "Yes", "No", "Yes", "No"],
            "TechSupport": ["No", "No", "Yes", "Yes", "Yes", "No"],
            "StreamingTV": ["No", "No", "Yes", "Yes", "Yes", "No"],
            "StreamingMovies": ["No", "No", "Yes", "No", "Yes", "No"],
        }
    )


def test_service_count_only_counts_yes():
    df = make_telco_sample()
    counts = build_service_count(df)
    assert counts.tolist() == [0, 2, 8, 4, 7, 0]


def test_select_and_scale_cluster_features():
    df = make_telco_sample()
    features = select_cluster_features(df)
    assert list(features.columns) == [
        "tenure",
        "MonthlyCharges",
        "service_count",
    ]

    scaled, scaler = scale_cluster_features(features)
    assert scaled.shape == (6, 3)
    assert np.allclose(scaled.mean(axis=0), 0.0)
    assert scaler.mean_.shape == (3,)


def test_candidate_evaluation_returns_kmeans_and_hierarchical():
    df = make_telco_sample()
    features = select_cluster_features(df)
    scaled, _ = scale_cluster_features(features)
    result = evaluate_cluster_candidates(scaled, k_values=range(2, 4))

    assert set(result["algorithm"]) == {"KMeans", "Hierarchical"}
    assert len(result) == 4
    assert result["silhouette"].notna().all()


def test_dbscan_evaluation_has_required_columns():
    df = make_telco_sample()
    features = select_cluster_features(df)
    scaled, _ = scale_cluster_features(features)
    result = evaluate_dbscan_candidates(
        scaled,
        eps_values=(0.5,),
        min_samples_values=(2, 3),
    )

    assert len(result) == 2
    assert set(["algorithm", "parameters", "silhouette", "noise_ratio"]).issubset(
        result.columns
    )
