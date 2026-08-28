"""Unit tests cho src/classification.py.

Kiểm tra đầy đủ và đồng đều cho cả Logistic Regression và Random Forest.
"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.classification import (
    build_classification_pipelines,
    predict_classifiers,
    tune_classifiers,
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


def test_tune_classifiers_structure(sample_data):
    X, y = sample_data
    tuned = tune_classifiers(X, y, cv=2)

    # Kiểm tra cả 2 mô hình đều có đầy đủ thông tin sau khi tune
    for model_name in ["logistic_regression", "random_forest"]:
        assert model_name in tuned
        assert "estimator" in tuned[model_name]
        assert "best_params" in tuned[model_name]
        assert "cv_score" in tuned[model_name]
        assert isinstance(tuned[model_name]["cv_score"], float)


def test_predict_classifiers_both_models(sample_data):
    X, y = sample_data
    tuned = tune_classifiers(X, y, cv=2)
    preds = predict_classifiers(tuned, X)

    # Kiểm tra kỹ dự đoán của Logistic Regression
    lr_pred = preds["logistic_regression"]
    assert isinstance(lr_pred, pd.Series)
    assert len(lr_pred) == len(X)
    assert set(lr_pred.unique()).issubset({0, 1})

    # Kiểm tra kỹ dự đoán của Random Forest (đầy đủ như Logistic Regression)
    rf_pred = preds["random_forest"]
    assert isinstance(rf_pred, pd.Series)
    assert len(rf_pred) == len(X)
    assert set(rf_pred.unique()).issubset({0, 1})

    # Kiểm tra tính năng dự đoán xác suất (predict_proba) của estimator Random Forest
    rf_estimator = tuned["random_forest"]["estimator"]
    rf_proba = rf_estimator.predict_proba(X)
    assert rf_proba.shape == (len(X), 2)