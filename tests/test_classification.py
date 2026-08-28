"""
Unit tests cho src/classification.py
"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.classification import (
    build_classification_pipelines,
    tune_classifiers,
    predict_classifiers,
)


@pytest.fixture
def sample_data():
    """Tạo dữ liệu giả lập chuẩn cấu trúc Telco Churn."""
    X = pd.DataFrame(
        {
            "tenure": [1, 12, 24, 60, 5, 36, 48, 2, 70, 10],
            "MonthlyCharges": [
                29.85,
                56.95,
                53.85,
                42.30,
                70.70,
                99.65,
                89.10,
                29.75,
                104.80,
                20.20,
            ],
            "TotalCharges": [
                29.85,
                1889.50,
                1081.45,
                1840.75,
                151.65,
                3204.40,
                3019.10,
                301.90,
                7261.25,
                202.00,
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
            ],
            "InternetService": [
                "DSL",
                "DSL",
                "DSL",
                "DSL",
                "Fiber optic",
                "Fiber optic",
                "Fiber optic",
                "DSL",
                "Fiber optic",
                "No",
            ],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 1, 0, 0, 0, 1], name="Churn")
    return X, y


def test_build_classification_pipelines(sample_data):
    X, _ = sample_data
    pipelines = build_classification_pipelines(X)

    assert "logistic_regression" in pipelines
    assert "random_forest" in pipelines
    assert isinstance(pipelines["logistic_regression"], Pipeline)
    assert isinstance(pipelines["random_forest"], Pipeline)


def test_tune_and_predict_classifiers(sample_data):
    X, y = sample_data
    tuned = tune_classifiers(X, y, cv=2)

    assert "logistic_regression" in tuned
    assert "random_forest" in tuned
    assert "estimator" in tuned["logistic_regression"]
    assert "best_params" in tuned["logistic_regression"]
    assert "cv_score" in tuned["logistic_regression"]

    preds = predict_classifiers(tuned, X)
    assert "logistic_regression" in preds
    assert "random_forest" in preds
    assert len(preds["logistic_regression"]) == len(X)
    assert set(preds["logistic_regression"].unique()).issubset({0, 1})